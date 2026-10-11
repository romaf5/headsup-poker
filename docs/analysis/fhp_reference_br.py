"""Exact best response (headsup.algos.holdem_br) to the SD-CFR average of a run of the authors' public code on FHP.

    PYTHONPATH=<DREAM>:<stubs>:. python docs/analysis/fhp_reference_br.py <run dir> <name> <T: first iterations of the bank> <cards: all | k> <device> <out.json>

(<DREAM>: github.com/EricSteinberger/DREAM with one compatibility patch - PokerRL/rl/neural/CardEmbedding.py moves the
range indices to the CPU before its numpy look-up; <stubs>: no-op stand-ins for gym / pycrayon / scipy.special.)

The authors' networks are queried through their own classes; the public observation of every decision node is taken
from their environment (the betting sequence replayed there) with the flop written into its board entries.  Checked
at start-up: both engines agree on the player to act, the legal actions and the pot at every decision node; the
constructed observation equals the environment's for a forced flop; hands map onto their range indices; and the
strategy computed here equals their ``IterationStrategy.get_a_probs2``.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch


def main():
    run_dir, name, T, cards, device, out_json = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    from PokerRL.game.games import Flop3Holdem

    Flop3Holdem.BIG_BET = int(os.environ.get("REF_FLOP_BET", "100"))  # as in run_fhp_ref.py
    from DREAM_and_DeepCFR.IterationStrategy import IterationStrategy
    from PokerRL.rl import rl_util

    from headsup.algos.holdem_br import MixturePolicy, VectorBestResponse, build_street_tree, parse_cards
    from headsup.engine import HeadsUpPoker
    from headsup.game import FHP

    dev = torch.device(device)
    with open(os.path.join(run_dir, "TrainingProfiles", name + ".pkl"), "rb") as f:
        t_prof = pickle.load(f)
    env_bldr = rl_util.get_env_builder(t_prof=t_prof)
    lut = env_bldr.lut_holder
    n_act = env_bldr.N_ACTIONS
    assert n_act == FHP.num_actions == 3

    # ---- cards: ours id = rank + 13 * suit; theirs via their look-up tables
    card_map = np.array([int(lut.LUT_2DCARD_2_1DCARD[c % 13, c // 13]) for c in range(52)])
    assert sorted(card_map) == list(range(52))
    card2d = np.asarray(lut.LUT_1DCARD_2_2DCARD)
    hole2idx = np.asarray(lut.LUT_HOLE_CARDS_2_IDX)
    idx2hole = np.asarray(lut.LUT_IDX_2_HOLE_CARDS)
    from headsup.lbr import COMBOS

    combos = np.asarray(COMBOS)
    th = card_map[combos]
    lo, hi = th.min(1), th.max(1)
    ridx_of_combo = hole2idx[lo, hi]
    assert len(set(ridx_of_combo.tolist())) == 1326 and (np.sort(idx2hole[ridx_of_combo], 1) == np.stack([lo, hi], 1)).all()
    card_map_t = torch.as_tensor(card_map, device=dev)
    card2d_t = torch.as_tensor(card2d, device=dev, dtype=torch.long)
    hole2idx_t = torch.as_tensor(hole2idx, device=dev, dtype=torch.long)
    bidx = torch.as_tensor(np.asarray(env_bldr.obs_board_idxs), device=dev, dtype=torch.long)
    assert len(bidx) == 9

    # ---- decision nodes of our tree -> their public observation
    def paths_of(nodes, prefix):
        paths = {0: prefix}
        for i, nd in enumerate(nodes):
            for a, c in nd.children.items():
                paths[c] = paths[i] + [a]
        return paths

    e = HeadsUpPoker(game=FHP)
    e.reset(list(range(9)))
    pre = build_street_tree(e)
    ppaths = paths_of(pre, [])
    records = [(nd, ppaths[i]) for i, nd in enumerate(pre) if nd.kind == 0 and nd.stage == 0]
    for i, nd in enumerate(pre):
        if nd.kind == 3:
            flop = build_street_tree(nd.engine)
            fpaths = paths_of(flop, ppaths[i])
            records += [(fn, fpaths[j]) for j, fn in enumerate(flop) if fn.kind == 0 and fn.stage == 1]

    def key_of(stage, hist):
        return (int(stage), tuple(np.round(np.asarray(hist, dtype=np.float64), 4).tolist()))

    wrapper = env_bldr.get_new_wrapper(is_evaluating=True)
    table = {}
    for nd, path in records:
        wrapper.reset()
        for a in path:
            _, _, done, _ = wrapper.step(a)
            assert not done
        env = wrapper.env
        assert env.current_player.seat_id == nd.player, (path, env.current_player.seat_id, nd.player)
        assert sorted(env.get_legal_actions()) == np.flatnonzero(nd.legal).tolist(), (path, env.get_legal_actions(), nd.legal)
        pot = env.main_pot + sum(s.current_bet for s in env.seats)
        assert pot == nd.engine.pot, (path, pot, nd.engine.pot)
        obs = np.asarray(wrapper.get_current_obs(), dtype=np.float32)
        k = key_of(nd.stage, nd.engine.observation()[31:79])
        assert k not in table, path
        table[k] = (torch.as_tensor(obs, device=dev), path)
    print(f"{len(table)} decision nodes: player to act, legal actions and pots agree in both engines", flush=True)

    def their_obs(tmpl, flops_ours):
        """tmpl (obs,), flops_ours (R, 3) our card ids -> (R, obs) with the flop written into the board entries."""
        obs = tmpl.expand(len(flops_ours), -1).clone()
        theirs = card_map_t[flops_ours]
        r2 = card2d_t[theirs]
        obs[:, bidx] = torch.stack([r2[..., 0], r2[..., 1], theirs], dim=-1).reshape(len(flops_ours), 9).to(obs.dtype)
        return obs

    # check: the constructed observation equals their environment's for a forced flop (after a limp and a check)
    wrapper.reset()
    wrapper.step(1)
    wrapper.step(1)
    env = wrapper.env
    flop_ours = np.array([3, 17, 45])
    env.board[:3] = card2d[card_map[flop_ours]]
    want = np.asarray(wrapper.get_current_obs(), dtype=np.float32)
    lim = HeadsUpPoker(game=FHP)
    lim.reset(list(range(9)))
    lim.step(1)
    lim.step(1)
    tmpl = table[key_of(1, lim.observation()[31:79])][0]
    got = their_obs(tmpl, torch.as_tensor(flop_ours[None], device=dev)).cpu().numpy()[0]
    assert np.array_equal(np.sort(got[bidx.cpu().numpy()].reshape(3, 3), 0), np.sort(want[bidx.cpu().numpy()].reshape(3, 3), 0)), (got, want)
    rest = np.setdiff1d(np.arange(len(want)), bidx.cpu().numpy())
    assert np.array_equal(got[rest], want[rest])
    print("board entries of the observation: as their environment writes them", flush=True)

    # ---- the bank
    strats = [[], []]
    for it in range(T):
        for p in (0, 1):
            with open(os.path.join(run_dir, "strategy_nets", name, f"{it}_P{p}.pkl.pkl"), "rb") as f:
                state = pickle.load(f)
            strats[p].append(IterationStrategy.build_from_state_dict(t_prof=t_prof, env_bldr=env_bldr, device=dev, state=state))
    weights = [s.iteration + 1 for s in strats[0]]  # StrategyBuffer.add
    assert [s.iteration for s in strats[0]] == list(range(T)) == [s.iteration for s in strats[1]]

    def probs_of(strat, obs, ridx_np, masks):
        """IterationStrategy.get_a_probs2 without its device round trip of the range indices (their card look-up indexes
        numpy arrays, so the indices stay on the CPU)."""
        if strat._adv_net is None:  # as their evaluation path (_get_a_probs_of_hands): uniform only without a network
            return masks / masks.sum(-1, keepdim=True)
        adv = strat._adv_net(pub_obses=obs, range_idxs=ridx_np, legal_action_masks=masks)
        relu = torch.relu(adv)
        tot = relu.sum(1, keepdim=True)
        best = torch.zeros_like(adv)
        best[torch.arange(len(adv), device=adv.device), torch.where(masks.bool(), adv, torch.full_like(adv, -10e20)).argmax(1)] = 1
        return torch.where(tot > 0, relu / tot, best)

    # check against their own code path (CPU copies of two entries)
    rng = np.random.default_rng(0)
    for s in [x for x in (strats[0][min(1, T - 1)], strats[1][T - 1]) if x.iteration > 0]:  # (get_a_probs2 is uniform at iteration 0)
        c = IterationStrategy.build_from_state_dict(t_prof=t_prof, env_bldr=env_bldr, device=torch.device("cpu"), state=s.state_dict())
        tmpl, path = list(table.values())[int(rng.integers(len(table)))]
        flops = torch.as_tensor(np.stack([rng.permutation(52)[:3] for _ in range(64)]), device=dev)
        obs = their_obs(tmpl, flops)
        ridx = rng.integers(0, 1326, 64)
        wrapper.reset()
        for a in path:
            wrapper.step(a)
        legal = wrapper.env.get_legal_actions()
        m = torch.zeros(64, n_act, device=dev)
        m[:, legal] = 1
        mine = probs_of(s, obs, ridx, m).cpu().numpy()
        theirs = c.get_a_probs2(pub_obses=obs.cpu(), range_idxs=ridx.tolist(), legal_action_masks=m.cpu(), to_np=True)
        assert np.allclose(mine, theirs, atol=1e-5), np.abs(mine - theirs).max()
    print("strategies: equal to IterationStrategy.get_a_probs2", flush=True)

    ridx_of_combo_t = torch.as_tensor(ridx_of_combo, device=dev, dtype=torch.long)

    def fn(seat, x, legal):
        stage = int(x[0, 21].item())
        tmpl, _ = table[key_of(stage, x[0, 31:79].cpu().numpy())]
        R = len(x)
        if stage >= 1:
            obs = their_obs(tmpl, (x[:, [8, 11, 14]] - 1).round().long())
        else:
            obs = tmpl.expand(R, -1)
        h = card_map_t[(x[:, [2, 5]] - 1).round().long()]
        ridx = hole2idx_t[h.min(1).values, h.max(1).values].cpu().numpy()
        masks = legal.to(torch.float32).expand(R, -1)
        return torch.stack([probs_of(s, obs, ridx, masks) for s in strats[seat]])

    pol = MixturePolicy(fn, weights, FHP, dev, max_rows=1 << 15)
    t0 = time.perf_counter()
    res = VectorBestResponse(pol, FHP, cards=parse_cards(cards), chunk=int(os.environ.get("REF_CHUNK", "16")), seed=0).run()
    res["seconds"] = time.perf_counter() - t0
    res["policy"] = f"authors' code: {run_dir}, the networks of the first {T} iterations with weights t"
    print(f"{name}@t{T}: exploitability {res['exploitability_mbb']:.1f} mbb/g (total {res['total_exploitability_mbb']:.1f}; "
          f"BR values {res['br_values'][0]:+.3f} / {res['br_values'][1]:+.3f}, profile values {res['values'][0]:+.3f} / "
          f"{res['values'][1]:+.3f}; cards {cards}) in {res['seconds']:.0f}s", flush=True)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    with torch.no_grad():
        main()
