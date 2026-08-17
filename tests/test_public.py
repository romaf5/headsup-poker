import numpy as np
import pytest

from headsup.engine import HeadsUpPoker
from headsup.game import DEFAULT_GAME, GameConfig
from headsup.public import board_cards, hero_cards, replay_from_obs

GAMES = [DEFAULT_GAME, GameConfig(bet_sizes=(0.5, 1.0, 2.0), mask_redundant=True), GameConfig(bet_sizes=("min", 1.0), raise_cap=4, mask_redundant=True)]


@pytest.mark.parametrize("game", GAMES)
def test_replay_from_obs_reproduces_the_public_state(game):
    rng = np.random.default_rng(7)
    e = HeadsUpPoker(rng=rng, game=game)
    checked = 0
    for _ in range(400):
        e.reset()
        while not e.done:
            obs = e.observation()
            hero = e.current
            seen = []
            e2, h2 = replay_from_obs(obs, game, on_action=lambda eng, seat, a: seen.append((seat, a, eng.pot)))
            assert h2 == hero and e2.current == hero and not e2.done
            for name in ("pot", "bets", "stage_bets", "stacks", "consecutive_raises", "history_n", "history_size", "acted"):
                assert getattr(e2, name) == getattr(e, name), name
            assert int(e2.stage) == int(e.stage)
            assert list(e2.visible_board) == (sorted(e.board[:3]) + list(e.board[3:]))[: len(e.visible_board)]
            assert sorted(e2.hands[hero]) == sorted(e.hands[hero]) and hero_cards(obs) == sorted(e.hands[hero])
            assert board_cards(obs) == list(e2.visible_board)
            np.testing.assert_array_equal(e2.observation(hero), obs)  # the hero's view is identical
            # the opponent's view differs only in the hand slots
            np.testing.assert_array_equal(e2.observation(1 - hero)[6:], e.observation(1 - hero)[6:])
            assert len(seen) == sum(e.history_n)
            assert e2.legal_mask() == e.legal_mask()
            checked += 1
            legal = np.flatnonzero(e.legal_mask())
            e.step(int(rng.choice(legal)))
    assert checked > 800
