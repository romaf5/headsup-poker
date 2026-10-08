"""AlphaHoldem's state tensors, rebuilt from the float32[80] observation (batched, on any torch device).

* Card tensor bool[6, 4, 13] (suit x rank; index 13 * suit + rank = the card id): the hole cards, the flop, the
  turn, the river, all public cards, hole + public cards.
* Action tensor bool[24, 4, A] (A = the game's number of actions): channel ``6 * round + slot`` is the slot-th
  decision of a betting round with the rows [seat 0's action, seat 1's action, their sum, the legal actions at
  that decision] (the paper's Fig. 3; seat 0 = small blind).  The decision the observer faces now has only its
  legal row, in the next free slot of the current round - which also tells whose turn it is.

The observation stores no actor, action index or legal set, only ``chips put in / pot before`` per slot
(obs[31:79]).  They follow from a replay from the blinds: heads-up actions alternate from the round's first actor
(seat 0 pre-flop, seat 1 after), so the pot, both seats' bets and stacks, the call amount and the raise counter
before every slot are cumulative products / sums over the slots (no loop over them).  The executed action is
check/call when the chips equal the call amount, fold when they are 0 facing a bet (terminal observations only),
all-in when they equal the stack, else the first raise size whose amount matches; the raise amounts come from a
table computed with the arithmetic of :meth:`GameConfig.raise_amount`, so the rounding is the engine's.

Limits: a round's 7th action is not in the observation; in a tree without ``mask_redundant`` a raise that the cap
or the stack turned into an all-in is recorded as the all-in, and of two sizes with the same amount the first.
The player has to be a function of the observation (``headsup.lbr`` substitutes hands in observations), which is
why the tensors are not written by the engine.
"""

import math

import numpy as np
import torch

from headsup.engine import HISTORY_OFFSET, HISTORY_ROUNDS, HISTORY_SLOTS, OBS_DIM_HISTORY, RAISES_INDEX, STACK_FEATURE_CAP
from headsup.game import DEFAULT_GAME

CARD_CHANNELS = 6
ACTION_CHANNELS = HISTORY_ROUNDS * HISTORY_SLOTS  # 24
ACTION_ROWS = 4  # seat 0, seat 1, sum, legal
SLOTS = ACTION_CHANNELS
_CARD_COLUMNS = (2, 5, 8, 11, 14, 17, 20)  # card id + 1 of the two hole cards and the five board cards


class Encoder:
    def __init__(self, game=DEFAULT_GAME, device="cpu"):
        if game.limit is None and game.stack_size > STACK_FEATURE_CAP:
            raise ValueError(f"the legal raises of a no-limit game with {game.stack_size}-chip stacks cannot be recovered from "
                             f"observations: the stack feature saturates at {STACK_FEATURE_CAP} chips")
        self.game = game
        self.device = torch.device(device)
        self.num_actions = game.num_actions
        # float64 where the device has it: the pot before a slot is a product of up to 24 float32 ratios
        self.ftype = torch.float32 if self.device.type == "mps" else torch.float64
        dev = self.device
        slot = torch.arange(SLOTS, device=dev)
        self.slot_round = slot // HISTORY_SLOTS
        # the seat acting in a slot: seat 0 opens the first round, seat 1 the others; then they alternate
        self.slot_seat0 = ((slot % HISTORY_SLOTS) + (self.slot_round > 0).long()) % 2 == 0
        self.caps = torch.tensor([game.cap(min(r, game.num_rounds - 1)) for r in range(HISTORY_ROUNDS)], device=dev)
        # chips of a pot-fraction raise beyond the call, by (raise, pot + call): the arithmetic of GameConfig.raise_amount
        # (a Python float product rounded half up), so no device rounds differently from the engines
        top = 3 * game.stack_size + 1 if game.limit is None else 1
        table = np.zeros((game.num_raises, top), dtype=np.int64)
        self.fraction = [size not in ("min", "limit") for size in game.bet_sizes]
        for k, size in enumerate(game.bet_sizes):
            if self.fraction[k]:
                table[k] = [int(math.floor(size * x + 0.5)) for x in range(top)]
        self.table = torch.as_tensor(table, device=dev)
        self.limit = torch.tensor(list(game.limit) + [0] * (HISTORY_ROUNDS - len(game.limit)), device=dev) if game.limit else None

    # ------------------------------------------------------------------ the engine's rules, vectorised
    def _raise_amounts(self, to_call, pot, stack, rnd):
        """int64[..., K]: :meth:`GameConfig.raise_amount` for every raise size."""
        g = self.game
        out = []
        for k in range(g.num_raises):
            if self.limit is not None:
                amount = to_call + self.limit[rnd]
            elif self.fraction[k]:
                x = (pot + to_call).clamp(0, self.table.shape[1] - 1)
                amount = torch.maximum(to_call + g.big_blind, to_call + self.table[k][x])
            else:
                amount = to_call + g.big_blind
            out.append(torch.minimum(amount, stack))
        return torch.stack(out, dim=-1)

    def _legal(self, to_call, pot, stack, raises, rnd):
        """(bool[..., A], int64[..., K]): :meth:`GameConfig.legal_mask` and the raise amounts, for int64 tensors of
        one shape."""
        g = self.game
        amounts = self._raise_amounts(to_call, pot, stack, rnd)
        mask = torch.ones(*to_call.shape, g.num_actions, dtype=torch.bool, device=to_call.device)
        mask[..., 0] = to_call > 0
        if g.mask_redundant or not g.all_in:
            cap = self.caps[rnd]
            dead = ((raises + 1 >= cap) if g.all_in else (raises >= cap)) | (stack <= to_call)
            for k in range(g.num_raises):
                bad = (dead | (amounts[..., k] >= stack)) if g.all_in else dead
                if g.mask_redundant:
                    for j in range(k):
                        bad = bad | (amounts[..., j] == amounts[..., k])
                mask[..., 2 + k] = ~bad
        return mask, amounts

    # ------------------------------------------------------------------ pieces of the observation
    def _obs(self, obs):
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _public(self, obs):
        """(pot, to_call, stack, raises, round) of the observed state, as :func:`headsup.engine.public_state_from_obs`."""
        pot = torch.round(obs[:, 29] * 1000)
        to_call = torch.round(obs[:, 23] * pot).long()
        stack = torch.round(obs[:, 28] * pot).long()
        raises = torch.round(obs[:, RAISES_INDEX]).long()
        stage = torch.round(obs[:, 21]).long()
        return pot.long(), to_call, stack, raises, stage

    def legal(self, obs):
        """bool[B, A]: the legal actions of the observed decision (= ``legal_mask_from_obs``)."""
        pot, to_call, stack, raises, stage = self._public(self._obs(obs))
        return self._legal(to_call, pot, stack, raises, stage.clamp(max=self.game.num_rounds - 1))[0]

    def value_bounds(self, obs):
        """(own, opp) float32[B]: the chips the observer and its opponent have put into the pot so far."""
        obs = self._obs(obs)
        pot = torch.round(obs[:, 29] * 1000)
        return torch.round(obs[:, 24] * pot), torch.round(obs[:, 25] * pot)

    def cards(self, obs):
        """bool[B, 6, 4, 13]."""
        obs = self._obs(obs)
        ids = torch.round(obs[:, _CARD_COLUMNS]).long() - 1  # -1: no card
        onehot = torch.zeros(len(obs), 7, 52, dtype=torch.bool, device=self.device)
        onehot.scatter_(2, ids.clamp(min=0).unsqueeze(-1), (ids >= 0).unsqueeze(-1))
        hole, flop, turn, river = onehot[:, 0] | onehot[:, 1], onehot[:, 2:5].any(dim=1), onehot[:, 5], onehot[:, 6]
        public = flop | turn | river
        return torch.stack([hole, flop, turn, river, public, hole | public], dim=1).view(len(obs), CARD_CHANNELS, 4, 13)

    def actions(self, obs):
        """(bool[B, 24, 4, A], bool[B, A]): the action tensor and the legal actions of the observed decision."""
        obs = self._obs(obs)
        g, n, B = self.game, self.num_actions, len(obs)
        hist = obs[:, HISTORY_OFFSET:OBS_DIM_HISTORY].reshape(B, SLOTS, 2)
        occurred = hist[..., 1] > 0.5
        # pot after every slot = blinds * prod(1 + chips / pot before); chips are integers, so rounding is exact
        growth = torch.cumprod(1.0 + hist[..., 0].to(self.ftype), dim=1)
        pot_after = torch.round((g.small_blind + g.big_blind) * growth).long()
        pot_before = torch.cat([torch.full_like(pot_after[:, :1], g.small_blind + g.big_blind), pot_after[:, :-1]], dim=1)
        amount = pot_after - pot_before
        seat0 = self.slot_seat0
        put0, put1 = amount * seat0, amount * ~seat0
        # chips of each seat before a slot: over the hand (-> stacks) and within the slot's round (-> the call amount)
        total0 = g.small_blind + put0.cumsum(1) - put0
        total1 = g.big_blind + put1.cumsum(1) - put1
        stack = g.stack_size - torch.where(seat0, total0, total1)
        r0, r1 = put0.view(B, HISTORY_ROUNDS, HISTORY_SLOTS), put1.view(B, HISTORY_ROUNDS, HISTORY_SLOTS)
        street0, street1 = r0.cumsum(2) - r0, r1.cumsum(2) - r1
        street0[:, 0] += g.small_blind
        street1[:, 0] += g.big_blind
        street0, street1 = street0.view(B, SLOTS), street1.view(B, SLOTS)
        to_call = torch.where(seat0, street1 - street0, street0 - street1)
        call = torch.minimum(to_call, stack)
        # raises before a slot in its round: once a round has a raise, only raises follow until it closes
        raised = (occurred & (amount > call)).view(B, HISTORY_ROUNDS, HISTORY_SLOTS).long()
        raises = (raised.cumsum(2) - raised).view(B, SLOTS)
        rnd = self.slot_round.expand(B, SLOTS).clamp(max=g.num_rounds - 1)
        legal_then, amounts = self._legal(to_call, pot_before, stack, raises, rnd)
        match = amounts == amount.unsqueeze(-1)
        action = torch.where(match.any(-1), match.long().argmax(-1) + 2, 1)  # the first raise size with that amount
        if g.all_in:
            action = torch.where(amount == stack, g.all_in_action, action)
        action = torch.where(amount == call, 1, action)
        action = torch.where((amount == 0) & (call > 0), 0, action)
        taken = torch.nn.functional.one_hot(action, n).bool() & occurred.unsqueeze(-1)
        out = torch.zeros(B, SLOTS, ACTION_ROWS, n, dtype=torch.bool, device=self.device)
        out[:, :, 0] = taken & seat0.view(1, SLOTS, 1)
        out[:, :, 1] = taken & ~seat0.view(1, SLOTS, 1)
        out[:, :, 2] = taken
        out[:, :, 3] = legal_then & occurred.unsqueeze(-1)
        # the decision the observer faces: its legal actions in the next free slot of the current round
        pot, to_call_now, stack_now, raises_now, stage = self._public(obs)
        legal_now = self._legal(to_call_now, pot, stack_now, raises_now, stage.clamp(max=g.num_rounds - 1))[0]
        r = stage.clamp(max=HISTORY_ROUNDS - 1)
        k = occurred.view(B, HISTORY_ROUNDS, HISTORY_SLOTS).sum(2).gather(1, r.unsqueeze(1)).squeeze(1)
        rows = torch.nonzero((stage < HISTORY_ROUNDS) & (k < HISTORY_SLOTS)).squeeze(1)
        out[rows, (r * HISTORY_SLOTS + k)[rows], 3] = legal_now[rows]
        return out, legal_now

    def __call__(self, obs):
        """(cards bool[B, 6, 4, 13], actions bool[B, 24, 4, A], legal bool[B, A])."""
        obs = self._obs(obs)
        return (self.cards(obs), *self.actions(obs))
