"""Card representation.

Cards are plain ints ``0..51`` with ``rank = card % 13`` (0 = deuce … 12 = ace) and
``suit = card // 13`` in the order spades, hearts, diamonds, clubs.  This is exactly the
``card_index`` the original observation processor derived from ``treys`` ints, so models
trained on the old code keep working.  Conversion tables to/from ``treys`` are precomputed.
"""

import numpy as np
from treys import Card as TreysCard
from treys import Evaluator as TreysEvaluator

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "shdc"  # spades, hearts, diamonds, clubs (treys suit ints 1, 2, 4, 8)
NUM_CARDS = 52


def card_from_str(text: str) -> int:
    """'Ah' -> card id."""
    rank = RANK_CHARS.index(text[0].upper())
    suit = SUIT_CHARS.index(text[1].lower())
    return rank + 13 * suit


def card_to_str(card: int) -> str:
    return RANK_CHARS[card % 13] + SUIT_CHARS[card // 13]


CARD_STRS = [card_to_str(c) for c in range(NUM_CARDS)]
TREYS_CARDS = [TreysCard.new(s) for s in CARD_STRS]

# Per-card observation features: (rank + 1, suit + 1, card + 1); 0 is reserved for "no card".
CARD_FEATURES = np.zeros((NUM_CARDS, 3), dtype=np.float32)
for _c in range(NUM_CARDS):
    CARD_FEATURES[_c] = (_c % 13 + 1, _c // 13 + 1, _c + 1)

_EVALUATOR = TreysEvaluator()


def hand_strength(hand, board) -> int:
    """treys hand rank of ``hand`` (2 cards) + ``board`` (3..5 cards); lower is stronger."""
    return _EVALUATOR.evaluate(
        [TREYS_CARDS[c] for c in hand], [TREYS_CARDS[c] for c in board]
    )


def hand_class_str(hand, board) -> str:
    rank = hand_strength(hand, board)
    return _EVALUATOR.class_to_string(_EVALUATOR.get_rank_class(rank))


_RANK_NAMES = ["deuce", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "jack", "queen", "king", "ace"]
_RANK_PLURAL = ["deuces", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines", "tens", "jacks", "queens", "kings", "aces"]


def best_five(hand, board):
    """The best 5-card combination (list of card ids) out of hand + board."""
    from itertools import combinations

    cards = list(hand) + list(board)
    if len(cards) <= 5:
        return cards
    best, best_rank = None, 10**9
    for combo in combinations(cards, 5):
        r = _EVALUATOR._five([TREYS_CARDS[c] for c in combo])
        if r < best_rank:
            best, best_rank = list(combo), r
    return best


def describe_hand(hand, board):
    """Human-readable strength, e.g. 'pair of kings', 'ace-high', 'eights full of fives'."""
    if len(board) < 3:
        return None
    five = best_five(hand, board)
    ranks = sorted((c % 13 for c in five), reverse=True)
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)  # by count, then rank
    cls = _EVALUATOR.get_rank_class(hand_strength(hand, board))
    name = _EVALUATOR.class_to_string(cls)
    top = groups[0][0]
    if name == "High Card":
        return f"{_RANK_NAMES[ranks[0]]}-{_RANK_NAMES[ranks[1]]} high"
    if name == "Pair":
        return f"pair of {_RANK_PLURAL[top]}"
    if name == "Two Pair":
        return f"two pair, {_RANK_PLURAL[groups[0][0]]} and {_RANK_PLURAL[groups[1][0]]}"
    if name == "Three of a Kind":
        return f"three {_RANK_PLURAL[top]}"
    if name in ("Straight", "Straight Flush", "Royal Flush"):
        high = ranks[0] if not (ranks[0] == 12 and ranks[1] == 3) else 3  # wheel: A-2-3-4-5
        label = "royal flush" if name == "Royal Flush" else name.lower()
        return f"{_RANK_NAMES[high]}-high {label}"
    if name == "Flush":
        return f"{_RANK_NAMES[ranks[0]]}-high flush"
    if name == "Full House":
        return f"{_RANK_PLURAL[groups[0][0]]} full of {_RANK_PLURAL[groups[1][0]]}"
    if name == "Four of a Kind":
        return f"four {_RANK_PLURAL[top]}"
    return name.lower()
