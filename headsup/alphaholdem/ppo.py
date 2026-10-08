"""Trinal-Clip PPO (AlphaHoldem, section "Effective Learning with Trinal-Clip PPO") and the advantages it is fed.

The paper's eq. (3), ``clip(r, clip(r, 1 - eps, 1 + eps), delta1) * A``, has no ``min`` and no sign condition; its
text applies delta1 "when A < 0" and cites the dual-clip PPO of Ye et al. (2020), whose loss for A < 0 is
``max(min(r A, clip(r, 1 - eps, 1 + eps) A), c A)``.  That reading is implemented: PPO's clipped surrogate, bounded
below by ``delta1 * A`` when A < 0 - which for A < 0 equals eq. (3), ``clamp(r, 1 - eps, delta1) * A``.  (A ``min``
over the three terms, as one re-implementation has it, is plain PPO: the delta1 term never is the smallest.)

The value target ``clip(R, -delta2, delta3)`` (eq. 4) uses per-state bounds: delta2 / delta3 are the chips the player
/ the opponent have put in up to the state (OpenHoldem: "the state value when the player folds / the opponent
folds"); the other reading, the chips of the finished hand, never clips a complete hand's return.

Advantages: a rollout of the two-seat env is a grid [step, table] of decisions by alternating seats.  Each (table,
seat) pair is one stream of decisions; the hand's reward arrives after the stream's last decision of the hand and
nothing is carried across hands.  A hand in which a seat never acted (the other seat open-folded) has no decision to
credit: its reward is dropped for that seat - the two-seat form of the one-seat envs' terminal observation after a
reset.  Rollouts end at hand boundaries, so there is no bootstrap value.
"""

import torch


def trinal_clip_objective(ratio, adv, eps=0.2, delta1=3.0):
    """Per-sample policy objective (to be maximised): PPO's ``min(r A, clip(r, 1 - eps, 1 + eps) A)``, and for
    A < 0 not below ``delta1 * A``."""
    surr = torch.minimum(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv)
    return torch.where(adv < 0, torch.maximum(surr, delta1 * adv), surr)


def clipped_value_target(ret, own, opp, reward_scale, clip=True):
    """``clip(R, -delta2, delta3)`` with delta2 = ``own`` and delta3 = ``opp`` chips put in at the state (returns
    are in units of ``reward_scale`` chips)."""
    if not clip:
        return ret
    return torch.maximum(torch.minimum(ret, opp / reward_scale), -own / reward_scale)


def stream_gae(values, rewards, dones, seats, active, gamma, lam):
    """GAE along the (table, seat) streams of a lock-step rollout.

    ``values`` float[T, N]: the value at each decision (anything where it is not needed); ``rewards`` float[T, N, 2]:
    both seats' rewards where ``dones`` (bool[T, N]: the step's action ended the hand); ``seats`` int[T, N]: the
    acting seat; ``active`` bool[T, N]: the table acted at that step (it did not wait).
    Returns ``(advantages, returns)`` float[T, N] (0 at waiting cells): the reward follows a stream's last decision
    of the hand undiscounted, earlier decisions see it through gamma per decision of that seat; the return is the
    lambda = 1 one.  Hands still open at the end count as unrewarded (the trainer's rollouts have none)."""
    T, N = values.shape
    adv, ret = torch.zeros_like(values), torch.zeros_like(values)
    zeros = values.new_zeros(N, 2)
    next_value, next_adv, next_ret, reward = zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()
    tables = torch.arange(N, device=values.device)
    for t in range(T - 1, -1, -1):
        act = active[t]
        ended = (dones[t] & act).unsqueeze(1)  # going backwards: the decisions before this step belong to this hand
        next_value, next_adv, next_ret = next_value.masked_fill(ended, 0.0), next_adv.masked_fill(ended, 0.0), next_ret.masked_fill(ended, 0.0)
        reward = torch.where(ended, rewards[t], reward)
        s = seats[t]
        r, v = reward[tables, s], values[t]
        a = r + gamma * next_value[tables, s] - v + gamma * lam * next_adv[tables, s]
        g = r + gamma * next_ret[tables, s]
        adv[t], ret[t] = torch.where(act, a, 0.0), torch.where(act, g, 0.0)
        # the acting seat's stream now continues from this decision; its reward is spent
        next_value[tables, s] = torch.where(act, v, next_value[tables, s])
        next_adv[tables, s] = torch.where(act, a, next_adv[tables, s])
        next_ret[tables, s] = torch.where(act, g, next_ret[tables, s])
        reward[tables, s] = torch.where(act, 0.0, reward[tables, s])
    return adv, ret


def ppo_loss(net, batch, eps, delta1, value_coef, entropy_coef, reward_scale, value_clip=True):
    """Total loss ``-policy + value_coef * value - entropy_coef * entropy`` of a minibatch and its statistics.

    ``batch``: ``cards``, ``acts``, ``legal`` (the network's inputs), ``action``, ``logp`` (log-probability under
    the policy that acted), ``adv``, ``ret``, ``own`` / ``opp`` (chips put in at the state: the value-clip bounds)."""
    logits, value = net(batch["cards"], batch["acts"], batch["legal"])
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
    ratio = torch.exp(logp - batch["logp"])
    adv = batch["adv"]
    policy = trinal_clip_objective(ratio, adv, eps, delta1).mean()
    target = clipped_value_target(batch["ret"], batch["own"], batch["opp"], reward_scale, value_clip)
    value_loss = ((target - value) ** 2).mean()
    entropy = -(logp_all.exp() * logp_all).sum(dim=1).mean()  # masked actions: probability 0 times a finite logit
    loss = -policy + value_coef * value_loss - entropy_coef * entropy
    with torch.no_grad():
        stats = {
            "policy": policy.item(), "value": value_loss.item(), "entropy": entropy.item(),
            "kl": (batch["logp"] - logp).mean().item(),
            "clipped": ((ratio - 1.0).abs() > eps).float().mean().item(),
            "delta1_clipped": ((adv < 0) & (ratio > delta1)).float().mean().item(),
            "value_clipped": (target != batch["ret"]).float().mean().item(),
        }
    return loss, stats
