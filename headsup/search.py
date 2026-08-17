"""Real-time search: depth-limited (current-street) subgame re-solving at play time.

Follows the "unsafe subgame solving" of Brown & Sandholm (2017, *Safe and Nested Subgame
Solving for Imperfect-Information Games*): at a decision the remaining game from the current
public state is solved with CFR, the root being a chance node that deals both players' hands
from their *ranges* - the reach probabilities of every hand under the blueprint (for the
opponent) and under the strategies actually played (for the hero) - and the result is played
for the hero's real hand ("nested": the process repeats at every later decision with the
updated ranges).  The subgame is depth-limited to the end of the current betting round as in
Brown, Sandholm & Amos (2018, *Depth-Limited Solving for Imperfect-Information Games*): at a
street-end leaf the hand is rolled out with a continuation strategy (the blueprint: the
DeepCFR policy net, the last SD-CFR iterate or a thinned iterate bank; several can be given
and are sampled per rollout, which is how Modicum / Pluribus estimate leaf values with a
handful of continuation strategies).  Before the river the solver is external-sampling MCCFR
with tabular regrets over (public sequence, hand) infosets and linear averaging (LCFR - with
sampled regrets it beat DCFR / CFR+ / PCFR+ in our tests), the hero's real hand being dealt on
half of its own traversals ("targeted" sampling); on the river every leaf is terminal and the
subgame is solved exactly by full-width vector-form CFR over all 1326 hands (default variant
DCFR, Brown & Sandholm 2019, alpha 1.5 / beta 0 / gamma 2; CFR+ and PCFR+ (predictive RM+,
Farina, Kroer & Sandholm 2021) available) - ~200 iterations reach ~0.01 chips per hand pair of
exploitability in well under a second.  Both solvers are C++ (``headsup_cpp.SubgameSolver`` /
``VectorSolver``); ``exploitability`` computes the exact best response of both players against
the solved strategies on river subgames - the correctness check.

Pluribus mode (``@pluribus``; Brown & Sandholm 2019, *Superhuman AI for multiplayer poker*,
supplementary material): the blueprint plays the first betting round; from the flop on every
decision re-solves the *whole remaining game* from the start of the current betting round
(``VectorSolver``: vector-form Linear CFR with public chance sampling, lossless hands in the
current round, ``buckets`` equity buckets on later rounds), the hero's actions already taken
in the round frozen for its real hand only, playing the final iterate's strategy; the ranges
at the start of a round come from Bayes' rule with the previous round's solve (its average
strategy, for both players - "nested unsafe search"), preflop with the blueprint.

Player spec: ``search:<blueprint spec>[@it<N>][@rit<N>][@rv<variant>][@focus<f>][@cont<policy|iterate|bank>][@thin<K>]``,
e.g. ``search:cfr:runs/x/policy.pth@it20000@rit200`` (``@k4``: Pluribus's four biased continuation
strategies chosen by both players at the depth-limited leaves instead of one sampled continuation);
Pluribus mode ``search:<blueprint>@pluribus[@it<N>][@b<buckets>][@th<threads>][@avg][@pfsearch]``.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from headsup.cards import NUM_CARDS, hand_strength
from headsup.game import DEFAULT_GAME
from headsup.lbr import COMBOS, NUM_COMBOS, substitute_hands, valid_combos

_CARD_COMBOS = [np.flatnonzero((COMBOS == c).any(axis=1)) for c in range(NUM_CARDS)]


# ----------------------------------------------------------------------------- evaluation
def _opponent_mass(reach, blockers=True):
    """M[h] = sum of ``reach`` over combos compatible with combo h (no shared card)."""
    total = reach.sum()
    per_card = np.array([reach[_CARD_COMBOS[c]].sum() for c in range(NUM_CARDS)])
    return total - per_card[COMBOS[:, 0]] - per_card[COMBOS[:, 1]] + reach


def _showdown_values(reach, strength):
    """u[h] = sum over compatible h' of reach[h'] * (+1 if h beats h', -1 if h' beats h) with
    ``strength`` = treys rank per combo (lower is stronger; +inf for combos blocked by the board)."""
    order = np.argsort(strength, kind="stable")
    s_sorted = strength[order]
    r_sorted = reach[order]
    # for each combo: mass of strictly weaker (larger rank) and strictly stronger combos
    cum = np.cumsum(r_sorted)
    total = cum[-1]
    # position of the first / last combo with the same strength
    first = np.searchsorted(s_sorted, s_sorted, side="left")
    last = np.searchsorted(s_sorted, s_sorted, side="right")
    stronger_sorted = np.where(first > 0, cum[np.maximum(first - 1, 0)], 0.0)
    weaker_sorted = total - cum[last - 1]
    stronger = np.empty(NUM_COMBOS)
    weaker = np.empty(NUM_COMBOS)
    stronger[order] = stronger_sorted
    weaker[order] = weaker_sorted
    # remove combos sharing a card with h (they cannot be held): per card, the same cumulative trick
    for c in range(NUM_CARDS):
        idx = _CARD_COMBOS[c]
        sub = strength[idx]
        o = np.argsort(sub, kind="stable")
        ss, rr = sub[o], reach[idx][o]
        cs = np.cumsum(rr)
        f = np.searchsorted(ss, ss, side="left")
        l = np.searchsorted(ss, ss, side="right")
        st = np.where(f > 0, cs[np.maximum(f - 1, 0)], 0.0)
        wk = cs[-1] - cs[l - 1]
        stronger[idx[o]] -= st
        weaker[idx[o]] -= wk
    # h itself was subtracted twice (once per card): it is neither weaker nor stronger, nothing to add back
    return weaker - stronger


def exploitability(tree, strategies, ranges, board, game=DEFAULT_GAME):
    """Exact best-response values in a river subgame (all leaves terminal).

    ``tree``: ``SubgameSolver.tree()``; ``strategies[node]``: (1326, A) average strategy at every
    decision node; ``ranges``: (2, 1326) reach weights; ``board``: the 5 board cards.  Returns
    ``(exploitability, br_values, values)``: mean best-response gain (chips per hand pair, i.e. how
    much each player could gain on average by best-responding), the two best-response values and
    the two values of the profile itself.
    """
    strength = np.full(NUM_COMBOS, np.inf)
    ok = valid_combos(board)
    for h in np.flatnonzero(ok):
        strength[h] = hand_strength([int(COMBOS[h, 0]), int(COMBOS[h, 1])], board)
    r = [np.where(ok, ranges[0], 0.0).astype(np.float64), np.where(ok, ranges[1], 0.0).astype(np.float64)]

    def values(node, p, reach_q, best_response):
        """counterfactual values of player p's hands at ``node`` given opponent reach ``reach_q``."""
        nd = tree[node]
        if nd["kind"] == 1:
            sign = 1.0 if nd["folder"] != p else -1.0
            return sign * nd["stake"] * _opponent_mass(reach_q)
        if nd["kind"] == 2:
            return nd["stake"] * _showdown_values(reach_q, strength)
        if nd["kind"] != 0:
            raise ValueError("exploitability needs a subgame whose leaves are all terminal (river)")
        acts = [a for a in range(len(nd["legal"])) if nd["legal"][a]]
        if nd["player"] == p:
            child_values = np.stack([values(nd["child"][a], p, reach_q, best_response) for a in acts])
            if best_response:
                return child_values.max(axis=0)
            sig = strategies[node][:, acts].T  # (n_acts, 1326)
            return (sig * child_values).sum(axis=0)
        sig = strategies[node]
        return sum(values(nd["child"][a], p, reach_q * sig[:, a], best_response) for a in acts)

    joint = float(_opponent_mass(r[1]) @ r[0])  # sum over compatible hand pairs of r0 * r1
    out_br, out_v = [], []
    for p in (0, 1):
        q = 1 - p
        br = float(r[p] @ values(0, p, r[q], True)) / joint
        v = float(r[p] @ values(0, p, r[q], False)) / joint
        out_br.append(br)
        out_v.append(v)
    return 0.5 * (out_br[0] + out_br[1]), out_br, out_v


# ----------------------------------------------------------------------------- the player
def _continuations(spec, device, kind, thin):
    """C++ continuation strategies for the rollouts beyond the depth limit: lists (nets0, nets1, rm, weights).

    ``kind``: ``policy`` (the DeepCFR average-strategy net; softmax), ``iterate`` (the last SD-CFR
    iterate's advantage nets; regret matching) or ``bank`` (``thin`` representative iterates of the
    bank with their averaging weights - several continuation strategies, sampled per rollout).
    """
    import torch

    from headsup import native
    from headsup.model import BaseModel, load_model
    from headsup.players import parse_sdcfr_spec

    k, _, arg = spec.partition(":")
    k = k.lower()
    if k in ("cfr", "policy", "torch"):
        if kind != "policy":
            raise ValueError("a policy-net blueprint only offers the 'policy' continuation")
        from headsup.paths import DEFAULT_POLICY_PATH

        m = native.make_model(load_model(arg or DEFAULT_POLICY_PATH).numpy_weights())
        return [m], [m], [False], [1.0]
    if k in ("sdcfr", "iterate"):
        path, _, gamma, iterations, _ = parse_sdcfr_spec(arg)
        data = torch.load(path, map_location="cpu", weights_only=True)
        T = data["T"] if iterations is None else min(iterations, data["T"])
        weights = np.arange(1, T + 1, dtype=np.float64) ** gamma
        if kind in ("policy", "iterate"):
            picks, w = [T - 1], [1.0]
        else:  # bank: representative iterates of equal-weight bins (see IterateBank.thin)
            n = max(1, min(thin, T))
            cum = np.cumsum(weights) / weights.sum()
            edges = np.arange(1, n + 1) / n
            picks = sorted(set(int(np.searchsorted(cum, e_)) for e_ in edges))
            picks = [min(p_, T - 1) for p_ in picks]
            bounds = [-1] + picks
            w = [float(weights[bounds[i] + 1 : bounds[i + 1] + 1].sum()) for i in range(len(picks))]
        nets = [[], []]
        for t in picks:
            for seat in (0, 1):
                m = BaseModel(config=data["config"])
                m.load_state_dict({key: v[t] for key, v in data["seats"][seat].items()})
                nets[seat].append(native.make_model(m.numpy_weights()))
        return nets[0], nets[1], [True] * len(picks), list(w)
    raise ValueError(f"no continuation strategy for blueprint spec {spec!r}")


class SearchPlayer:
    """Plays a blueprint improved by real-time subgame search (see the module docstring).

    ``blueprint``: player spec of the strategy that (a) models the opponent for the range updates,
    (b) plays beyond the depth limit (``continuation``: policy | iterate | bank).  Batched and
    stateful (``ids``): per table it keeps both ranges and the number of history actions already
    accounted for; the public state is rebuilt from the observation at every decision.
    """

    wants_ids = True

    def __init__(self, blueprint, iterations=None, focus=0.5, continuation=None, thin=8, device=None, seed=0,
                 workers=None, game=None, river_iterations=200, river_variant="dcfr", warm_start=5000,
                 mode="depth", buckets=500, threads=4, play="final", preflop="blueprint", leaf_choices=1):
        from headsup import native
        from headsup.lbr import _model_for
        from headsup.players import make_player

        if mode not in ("depth", "pluribus"):
            raise ValueError("mode must be 'depth' or 'pluribus'")
        self.spec = blueprint
        self.mode = mode
        self.buckets, self.threads, self.play, self.preflop = int(buckets), int(threads), play, preflop
        self.leaf_choices = int(leaf_choices)  # depth-limited leaves: 1 sampled continuation, or Pluribus's 4 biased choices
        # depth mode: sampled MCCFR iterations (20k, ~0.2 s); Pluribus mode: vector iterations of the
        # whole remaining game (500: ~2-3 s on the flop with 16 threads, well under a second later)
        self.iterations = int(iterations) if iterations is not None else (20_000 if mode == "depth" else 500)
        self.focus = float(focus)
        self.river_iterations, self.river_variant = int(river_iterations), river_variant
        self.warm_start = int(warm_start)  # equivalent iterations of the previous solve's regrets to start from (0 = off)
        self.model = _model_for(blueprint, device, seed + 1, model_iterates=thin, game=game)  # models the opponent
        self.model_observes = hasattr(self.model, "observe")
        self.game = getattr(self.model, "game", None) or make_player(blueprint, device=device, seed=seed).game
        kind = continuation or ("policy" if blueprint.split(":")[0] in ("cfr", "policy", "torch") else "iterate")
        self.continuation = kind
        self.conts = _continuations(blueprint, device, kind, thin) if (mode == "depth" or preflop == "search") else None
        self._cpp = native.module()
        self._cfg = native.engine_config(game=self.game)
        self.rng = np.random.default_rng(seed)
        self._pool = ThreadPoolExecutor(workers or 16)
        self.state = {}  # table id -> dict(villain, hero, processed, last_root)
        self.last_probs = None
        self.solve_time = 0.0
        self.solves = 0

    # ------------------------------------------------------------------ per-table bookkeeping
    def _fresh(self, obs):
        from headsup.public import hero_cards

        mine = hero_cards(obs)
        villain = valid_combos(mine).astype(np.float64)
        hero = np.ones(NUM_COMBOS)
        return {"villain": villain, "hero": hero, "processed": 0, "last_root": None, "last_action": None, "cards": tuple(mine),
                "solver": None, "solver_actions": None, "solver_round": -1}

    @staticmethod
    def _n_actions(obs):
        from headsup.engine import HISTORY_ROUNDS, HISTORY_SLOTS, history_slot

        return int(sum(obs[history_slot(r, k) + 1] > 0 for r in range(HISTORY_ROUNDS) for k in range(HISTORY_SLOTS)))

    def _prepare(self, obs_row, tid):
        """Rebuild the public state; return (engine, hero_seat, actions, pending villain decisions)."""
        from headsup.public import hero_cards, replay_from_obs

        st = self.state.get(tid)
        n_now = self._n_actions(obs_row)
        if st is None or n_now < st["processed"] or st["cards"] != tuple(hero_cards(obs_row)):
            st = self.state[tid] = self._fresh(obs_row)
        actions, pending, hero_pending = [], [], []
        counter = [0]

        rounds = []

        def on_action(engine, seat, action):
            i = counter[0]
            counter[0] += 1
            actions.append(action)
            rounds.append(int(engine.stage))
            if i < st["processed"]:
                return
            if seat != int(obs_row[22]):  # villain decision: needs the blueprint's strategy for every hand
                pending.append((engine.observation(seat), engine.clone(), action, i))
            else:  # hero decision taken by us (or by someone else: fall back to the blueprint)
                hero_pending.append((engine.observation(seat), engine.clone(), action, i))

        engine, hero = replay_from_obs(obs_row, self.game, on_action=on_action)
        st["rounds"] = rounds
        return engine, hero, actions, pending, hero_pending, st

    def _solved_strategy(self, st, index, actions):
        """The average strategy of the round's stored solve at the node of action ``index`` (all
        hands, root-round node), or None when no solve covers it."""
        sv, prefix = st.get("solver"), st.get("solver_actions")
        if sv is None or prefix is None or st.get("solver_round") != st["rounds"][index] or actions[: len(prefix)] != prefix:
            return None
        node = 0
        for a in actions[len(prefix): index]:
            node = sv.child(node, int(a))
            if node <= 0:
                return None
        if sv.node_player(node) < 0 or sv.node_round(node) != sv.root_round:
            return None
        return sv.node_strategy(node).astype(np.float64)

    def _update_ranges(self, jobs):
        """Apply the pending villain (blueprint) and hero (own solved / blueprint) range updates.

        Pluribus mode: only actions of *earlier* betting rounds are applied (the current round's
        actions are inside the solved tree, whose root is the round start), using the average
        strategy of that round's solve for both players when there was one (nested unsafe
        search), the blueprint otherwise (the first betting round).
        """
        from headsup.lbr import transition_likelihood

        queries = []  # (table id, kind, obs, engine, action)
        for tid, (engine, hero, actions, pending, hero_pending, st) in jobs.items():
            if self.mode == "pluribus":
                current = int(engine.stage)
                keep = [x for x in pending + hero_pending if st["rounds"][x[3]] < current or engine.done]
                keep.sort(key=lambda x: x[3])
                for obs_x, eng, a, i in keep:
                    kind = "villain" if int(obs_x[22]) != hero else "hero"
                    sig = self._solved_strategy(st, i, actions)
                    if sig is not None:
                        st[kind] *= transition_likelihood(eng, a, sig)
                    else:
                        queries.append((tid, kind, obs_x, eng, a))
                st["processed_target"] = max([x[3] + 1 for x in keep], default=st["processed"])
                continue
            for obs_v, eng, a, _ in pending:
                queries.append((tid, "villain", obs_v, eng, a))
            for obs_h, eng, a, _ in hero_pending:
                if st["last_root"] is not None and st["last_action"] == a and st["last_root"].shape[0] == NUM_COMBOS:
                    st["hero"] *= transition_likelihood(eng, a, st["last_root"])
                    st["last_root"] = None
                else:
                    queries.append((tid, "hero", obs_h, eng, a))
        if queries:
            rows = np.concatenate([substitute_hands(q[2]) for q in queries])
            ids = np.concatenate([q[0] * NUM_COMBOS + np.arange(NUM_COMBOS) for q in queries])
            probs = np.asarray(self.model.probs(rows, ids), dtype=np.float64).reshape(len(queries), NUM_COMBOS, -1)
            if self.model_observes:
                self.model.observe(rows, ids, np.repeat([q[4] for q in queries], NUM_COMBOS))
            for (tid, kind, _, eng, a), sig in zip(queries, probs):
                jobs[tid][5][kind] *= transition_likelihood(eng, a, sig)
        for tid, (engine, hero, actions, pending, hero_pending, st) in jobs.items():
            st["processed"] = st.pop("processed_target", len(actions)) if self.mode == "pluribus" else len(actions)
            board = list(engine.visible_board)
            if board:
                ok = valid_combos(board)
                st["villain"] *= ok
                st["hero"] *= ok
            for key in ("villain", "hero"):
                total = st[key].sum()
                if total <= 1e-12:  # the model gave the observed line probability 0: fall back to uniform
                    st[key] = valid_combos(board + (list(st["cards"]) if key == "villain" else [])).astype(np.float64)
                    total = st[key].sum()
                st[key] /= total

    def _solve_round(self, engine, hero, actions, st, seed):
        """Pluribus mode: solve the remaining game from the start of the current betting round
        (the hero's actions taken in the round frozen for its real hand) and return the strategy
        at the hero's node (final iterate or average) plus the hero's combo index."""
        cpp = self._cpp
        current = int(engine.stage)
        n_root = next((i for i, r in enumerate(st["rounds"]) if r == current), len(actions))
        e = cpp.Engine(self._cfg)
        e.reset(list(engine.hands[0]) + list(engine.hands[1]) + list(engine.board))
        for a in actions[:n_root]:
            e.step(int(a))
        ranges = [st["hero"], st["villain"]] if hero == 0 else [st["villain"], st["hero"]]
        a, b = sorted(st["cards"])
        hh = cpp.combo_index(int(a), int(b))
        sv = cpp.VectorSolver()
        sv.build(e, self.buckets)
        sv.set_ranges(ranges[0].astype(np.float32), ranges[1].astype(np.float32))
        node = 0
        for act in actions[n_root:]:
            if sv.node_player(node) == hero:
                sv.freeze(node, hh, int(act))
            node = sv.child(node, int(act))
        sv.run(self.iterations, int(seed) & 0xFFFFFFFF, self.threads)
        self.solves += 1
        st["solver"], st["solver_actions"], st["solver_round"] = sv, list(actions[:n_root]), current
        return sv.node_strategy(node, self.play == "final"), hh

    def _solve(self, engine, hero, actions, st, seed):
        cpp = self._cpp
        if self.mode == "pluribus" and (int(engine.stage) > 0 or self.preflop == "blueprint"):
            if int(engine.stage) == 0:  # the blueprint plays the first round
                rows = substitute_hands(engine.observation(hero))
                a, b = sorted(st["cards"])
                hh = cpp.combo_index(int(a), int(b))
                st["blueprint_rows"] = rows
                return None, hh
            return self._solve_round(engine, hero, actions, st, seed)
        e = cpp.Engine(self._cfg)
        e.reset(list(engine.hands[0]) + list(engine.hands[1]) + list(engine.board))
        for a in actions:
            e.step(int(a))
        ranges = [st["hero"], st["villain"]] if hero == 0 else [st["villain"], st["hero"]]
        a, b = sorted(st["cards"])
        hh = cpp.combo_index(int(a), int(b))
        if e.stage == 3:  # river: exact full-width solve
            sv = cpp.VectorSolver()
            sv.build(e)
            sv.set_ranges(ranges[0].astype(np.float32), ranges[1].astype(np.float32))
            sv.set_variant(self.river_variant)
            sv.run(self.river_iterations)
            return sv.root_strategy(), hh
        sv = cpp.SubgameSolver()
        sv.build(e)
        sv.set_ranges(ranges[0].astype(np.float32), ranges[1].astype(np.float32))
        n0, n1, rm, w = self.conts
        sv.set_continuations(n0, n1, rm, w)
        if self.leaf_choices > 1:
            sv.set_leaf_choices(self.leaf_choices)
        prev, prev_actions = st.get("solver"), st.get("solver_actions")
        if self.warm_start and prev is not None and prev_actions is not None and actions[: len(prev_actions)] == prev_actions:
            node = 0  # the new root inside the previous tree, if the street did not change
            for a in actions[len(prev_actions):]:
                node = prev.child(node, int(a)) if node >= 0 else -1
            if node > 0:
                try:
                    sv.warm_start(prev, node, self.warm_start)
                except RuntimeError:  # different street / structure: solve from scratch
                    pass
        sv.run(self.iterations, int(seed), hero, hh, self.focus)
        st["solver"], st["solver_actions"] = sv, list(actions)
        return sv.root_strategy(), hh

    # ------------------------------------------------------------------ player protocol
    def probs(self, obs, ids=None):
        from headsup.engine import legal_mask_from_obs

        obs = np.asarray(obs, dtype=np.float32)
        ids = np.arange(len(obs)) if ids is None else np.asarray(ids)
        jobs = {int(t): self._prepare(o, int(t)) for o, t in zip(obs, ids)}
        self._update_ranges(jobs)
        t0 = time.perf_counter()
        seeds = self.rng.integers(2**63, size=len(ids))
        # a finished hand (the opponent open-folded during the env's reset): any action is accepted
        futs = {int(t): self._pool.submit(self._solve, *jobs[int(t)][:3], jobs[int(t)][5], sd)
                for t, sd in zip(ids, seeds) if not jobs[int(t)][0].done}
        out = np.zeros((len(obs), self.game.num_actions), dtype=np.float32)
        blueprint = []  # (row index, table id, hh): preflop decisions the blueprint plays (Pluribus mode)
        for i, t in enumerate(ids):
            if int(t) not in futs:
                out[i, 1] = 1.0
                continue
            root, hh = futs[int(t)].result()
            if root is None:
                blueprint.append((i, int(t), hh))
                continue
            jobs[int(t)][5]["last_root"] = root.astype(np.float64)
            out[i] = root[hh]
        if blueprint:
            rows = np.concatenate([jobs[t][5].pop("blueprint_rows") for _, t, _ in blueprint])
            bids = np.concatenate([t * NUM_COMBOS + np.arange(NUM_COMBOS) for _, t, _ in blueprint])
            probs = np.asarray(self.model.probs(rows, bids), dtype=np.float32).reshape(len(blueprint), NUM_COMBOS, -1)
            for (i, t, hh), sig in zip(blueprint, probs):
                out[i] = sig[hh]
                jobs[t][5]["last_root"] = None
        self.solve_time += time.perf_counter() - t0
        legal = legal_mask_from_obs(obs, self.game)
        out[~legal] = 0.0
        s = out.sum(axis=1, keepdims=True)
        out = np.where(s > 0, out / np.maximum(s, 1e-12), legal / legal.sum(axis=1, keepdims=True))
        return out.astype(np.float32)

    def __call__(self, obs, ids=None):
        from headsup.players import sample_actions

        ids = np.arange(len(obs)) if ids is None else np.asarray(ids)
        self.last_probs = self.probs(obs, ids)
        actions = sample_actions(self.last_probs, self.rng)
        for t, a in zip(ids, actions):
            self.state[int(t)]["last_action"] = int(a)
        return actions


def parse_search_spec(arg):
    """``<blueprint spec>[@it<N>][@rit<N>][@rv<variant>][@focus<f>][@cont<policy|iterate|bank>][@thin<K>]``
    ``[@pluribus][@b<buckets>][@th<threads>][@avg][@pfsearch][@k<leaf choices>]`` -> kwargs."""
    parts = arg.split("@")
    # the blueprint spec itself may contain '@' options (sdcfr:...@g2): the search options are the
    # trailing ones that parse as ours; when an option repeats, the rightmost wins
    kw = {}
    while len(parts) > 1:
        o = parts[-1]
        if o.startswith("it") and o[2:].isdigit():
            kw.setdefault("iterations", int(o[2:]))
        elif o.startswith("rit") and o[3:].isdigit():
            kw.setdefault("river_iterations", int(o[3:]))
        elif o.startswith("rv") and o[2:] in ("lcfr", "dcfr", "cfr+", "pcfr+"):
            kw.setdefault("river_variant", o[2:])
        elif o.startswith("warm") and o[4:].isdigit():
            kw.setdefault("warm_start", int(o[4:]))
        elif o.startswith("focus"):
            kw.setdefault("focus", float(o[5:]))
        elif o.startswith("cont"):
            kw.setdefault("continuation", o[4:])
        elif o.startswith("thin") and o[4:].isdigit():
            kw.setdefault("thin", int(o[4:]))
        elif o == "pluribus":
            kw.setdefault("mode", "pluribus")
        elif o.startswith("b") and o[1:].isdigit():
            kw.setdefault("buckets", int(o[1:]))
        elif o.startswith("th") and o[2:].isdigit():
            kw.setdefault("threads", int(o[2:]))
        elif o == "avg":
            kw.setdefault("play", "average")
        elif o == "pfsearch":
            kw.setdefault("preflop", "search")
        elif o.startswith("k") and o[1:].isdigit():
            kw.setdefault("leaf_choices", int(o[1:]))
        else:
            break
        parts.pop()
    return "@".join(parts), kw
