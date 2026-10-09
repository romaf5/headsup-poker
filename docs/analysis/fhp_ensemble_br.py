"""Exact best response against the mixture of several policy nets (each hand is played by one of them, chosen with equal
probability): if the nets' errors are independent, the mixture is less exploitable than each of them.
usage: ensemble_br.py <cards: all|N> <policy.pth> <policy.pth> ..."""
import sys, torch
from headsup.algos.holdem_br import MixturePolicy, VectorBestResponse, _masked_softmax, parse_cards
from headsup.game import FHP
from headsup.model import load_model

dev = torch.device("cuda:0")
cards = parse_cards(sys.argv[1])
models = [load_model(p).to(dev).eval() for p in sys.argv[2:]]
fn = lambda seat, x, legal: torch.stack([_masked_softmax(m(x), legal) for m in models])
res = VectorBestResponse(MixturePolicy(fn, [1.0 / len(models)] * len(models), FHP, dev), FHP, cards=cards, chunk=32).run()
print(f"mixture of {len(models)}: total exploitability {res['total_exploitability_mbb']:.1f} mbb/g  (BR values {res['br_values']}, values {res['values']})")
