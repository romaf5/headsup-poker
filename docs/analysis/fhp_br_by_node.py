"""Where does an FHP strategy lose?  The exploitability when the best responder may deviate at one kind of decision node
only (a betting sequence of one street; the policy is played everywhere else), on 2,000 sampled flops.
usage: fhp_br_by_node.py <player spec> [cards]"""
import sys, torch
from headsup.algos.holdem_br import VectorBestResponse, mixture_policy, parse_cards
from headsup.game import FHP
from headsup.players import make_player

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cards = parse_cards(sys.argv[2]) if len(sys.argv) > 2 else 2000
policy = mixture_policy(make_player(sys.argv[1], device=dev, game=FHP), dev, FHP)

def key(node):
    """(street, actions made on it, raises among them): pre-flop '' / c / r / cr / rr / ...; the big blind counts as a bet."""
    e = node.engine
    return int(e.stage), int(e.history_n[int(e.stage)]), int(e.consecutive_raises)

def name(k):
    stage, n, r = k
    seq = "c" * (n - r) + "r" * r
    who = ("SB", "BB")[(n + stage) % 2]  # pre-flop the small blind opens, the flop the big blind
    return f"{('pre-flop', 'flop')[stage]} {who} after '{seq}'"

KEYS = [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 2), (0, 3, 2), (0, 3, 3), (0, 4, 3),
        (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 1), (1, 2, 2), (1, 3, 2), (1, 3, 3), (1, 4, 3), (1, 4, 4), (1, 5, 4)]
total = 0.0
for k in KEYS:
    r = VectorBestResponse(policy, FHP, cards=cards, chunk=32, br_filter=lambda nd, k=k: key(nd) == k).run()
    total += r["total_exploitability_mbb"]
    print(f"{name(k):32s} {r['total_exploitability_mbb']:6.1f} mbb/g", flush=True)
print(f"{'sum of the single-node gains':32s} {total:6.1f}")
