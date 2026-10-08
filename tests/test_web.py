import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from headsup.web.server import SessionHolder, make_handler
from headsup.web.session import GameSession, parse_action


def play_out(session):
    while not session.hand_over:
        if session.bot_to_act:
            session.bot_step()
        else:
            session.act("call")


def test_session_flow_and_bookkeeping():
    s = GameSession(opponent="call", advisor="cfr", device="cpu", seed=3)
    st = s.state()
    assert st["hand_number"] == 1 and st["you"]["position"] == "dealer" and st["your_turn"]
    assert [a["action"] for a in st["actions"]] == ["fold", "call", "raise", "allin"]  # SB faces the big blind
    assert st["actions"][1]["label"] == "Call 1" and st["actions"][2]["label"] == "Raise to 4"
    adv = s.advice()
    assert adv["probs"] is not None and abs(sum(p["p"] for p in adv["probs"]) - 1) < 1e-4
    assert 0 <= adv["equity"]["win"] <= 1
    play_out(s)
    assert s.hand_over and s.result is not None and len(s.results) == 1
    assert s.log and s.log[0]["seat"] == "you"
    with pytest.raises(ValueError):
        s.act("call")
    s.next_hand()
    st = s.state()
    assert st["hand_number"] == 2 and st["you"]["position"] == "big blind" and st["bot_to_act"]
    s.bot_step()
    assert s.bot_last is not None and s.bot_last["hand"] == 2
    for _ in range(3):
        while not s.hand_over:
            s.agent_step()
        s.next_hand()
    assert s.stats()["hands"] >= 4
    json.dumps(s.state())  # serialisable


def test_simple_bot_as_advisor_autoplays():
    s = GameSession(opponent="call", advisor="allin", device="cpu", seed=4)
    for _ in range(3):
        while not s.hand_over:
            s.agent_step()
        s.next_hand()
    assert s.stats()["hands"] == 3
    adv = s.advice() if s.state()["your_turn"] else None
    if adv:
        assert adv["probs"][3]["p"] == 1.0  # all-in advisor


def test_parse_action_and_labels():
    assert parse_action("fold") == 0 and parse_action("Check") == 1 and parse_action("all-in") == 3 and parse_action(2) == 2
    with pytest.raises(ValueError):
        parse_action("limp")
    s = GameSession(opponent="call", advisor=None, device="cpu", seed=1)
    s.act("call")  # limp
    s.bot_step()  # bot checks -> flop, bot acts first post-flop
    s.bot_step()  # bot checks
    labels = [a["label"] for a in s.state()["actions"]]
    assert labels == ["Check", "Bet 2", "All-in 98"]  # no fold when nothing is to call


def test_http_api_roundtrip():
    holder = SessionHolder(GameSession(opponent="call", advisor=None, device="cpu", seed=5))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(holder))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        def get(path):
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, r.read()

        def post(path, body=None):
            conn.request("POST", path, body=json.dumps(body or {}), headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            return r.status, json.loads(r.read())

        status, body = get("/")
        assert status == 200 and b"<title>" in body
        status, body = get("/static/app.js")
        assert status == 200
        status, st = post("/api/action", {"action": "raise"})
        assert status == 200 and st["bot_to_act"]
        status, st = post("/api/bot_step")
        assert status == 200
        status, st = post("/api/action", {"action": "nonsense"})
        assert status == 400 and "error" in st
        status, st = post("/api/new", {"opponent": "random", "stack_size": 50, "seed": 2})
        assert status == 200 and st["settings"]["stack_size"] == 50 and st["bot"]["name"] == "random"
        status, body = get("/api/players")
        assert status == 200 and json.loads(body)["players"][0]["spec"] == "cfr"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_table_follows_the_bots_trained_game_and_warns_when_overridden(tmp_path):
    """By default the table uses the stacks, blinds and raise cap the bot was trained with (it used fixed
    defaults, silently seating e.g. a raise-cap-4 bot at a raise-cap-3 table); an explicit override is reported."""
    from headsup.game import GameConfig
    from headsup.model import BaseModel

    path = tmp_path / "cap4.pth"
    BaseModel(game=GameConfig(raise_cap=4, stack_size=200, small_blind=5, big_blind=10)).save(path)
    s = GameSession(opponent=f"cfr:{path}", advisor=None, seed=0)
    assert (s.game.raise_cap, s.game.stack_size, s.game.small_blind, s.game.big_blind) == (4, 200, 5, 10)
    assert s.settings["raise_cap"] == 4 and s.settings["stack_size"] == 200 and s.state()["warning"] is None
    s = GameSession(opponent=f"cfr:{path}", advisor=None, seed=0, raise_cap=3)
    assert s.game.raise_cap == 3 and "raise cap 4" in s.state()["warning"]
    assert GameSession(opponent="call", advisor=None, seed=0).state()["warning"] is None  # bots without a trained tree


def test_limit_game_models_are_refused_with_a_clear_message(tmp_path):
    from headsup.game import FHP
    from headsup.model import BaseModel

    path = tmp_path / "fhp.pth"
    BaseModel(features="history", arch="paper", game=FHP).save(path)
    with pytest.raises(ValueError, match="limit games are not supported"):
        GameSession(opponent=f"cfr:{path}", advisor=None, seed=0)


def test_fold_with_nothing_to_call_is_logged_as_the_check_it_is():
    s = GameSession(opponent="call", advisor=None, seed=0)
    play_out(s)
    s.next_hand()  # the human is the big blind now: the calling bot limps first
    while s.bot_to_act:
        s.bot_step()
    s.act("fold")  # nothing to call: the engine executes a check
    mine = [entry for entry in s.log if entry["seat"] == "you"][-1]
    assert not s.hand_over and mine["text"] == "checks"


def test_allin_hands_report_equity_and_expected_result():
    """An all-in called before the river: the result also says what the hand was worth on average over the cards
    to come, and the session keeps an EV-adjusted total next to the dealt one."""
    s = GameSession(opponent="call", advisor=None, device="cpu", seed=5)
    assert s.state()["you"]["position"] == "dealer"
    s.act("allin")  # the calling station calls: 100 chips each, five cards to come
    while s.bot_to_act:
        s.bot_step()
    assert s.hand_over and s.result["showdown"]
    allin = s.result["allin"]
    assert allin["street"] == "pre-flop" and 0.0 < allin["equity"] < 1.0
    assert abs(allin["ev"]) < 100 and allin["ev"] == pytest.approx(100 * (2 * allin["equity"] - 1), abs=1e-6)
    assert allin["luck"] == pytest.approx(s.result["reward"] - allin["ev"])
    assert "equity" in allin["text"] and "%" in allin["text"]
    st = s.stats()
    assert st["ev_total"] == pytest.approx(allin["ev"]) and st["luck"] == pytest.approx(st["total"] - st["adj_total"])
    assert s.result["vs_range"]["ev"] != pytest.approx(allin["ev"])  # the calling station could hold anything: another average
    assert s.hand_records[-1]["ev"] == pytest.approx(allin["ev"])
    s.next_hand()  # big blind now; the calling station limps, we check it down: nothing was left to chance
    while not s.hand_over:
        if s.bot_to_act:
            s.bot_step()
        else:
            s.act("call")
    assert s.result["allin"] is None and s.hand_records[-1]["ev"] == s.result["reward"]
    st = s.stats()
    assert st["hands"] == 2 and st["ev_total"] == pytest.approx(allin["ev"] + s.result["reward"])
    assert len(st["cumulative_ev"]) == 2 and st["ev_mbb"] == pytest.approx(st["ev_total"] / 2 * 500)
    json.dumps(s.state())


def _showdown_ev_against(weights, me, board, stake):
    """Brute force: my expected result at a river showdown against opponent hands weighted by ``weights``."""
    from headsup.cards import hand_strength
    from headsup.lbr import COMBOS

    mine = hand_strength(me, board)
    total = ev = 0.0
    for h, w in enumerate(weights):
        a, b = int(COMBOS[h, 0]), int(COMBOS[h, 1])
        if w <= 0 or {a, b} & (set(me) | set(board)):
            continue
        s = hand_strength([a, b], board)
        ev += w * stake * ((mine < s) - (mine > s))
        total += w
    return ev / total


def test_showdowns_are_also_valued_against_the_bots_whole_range():
    """Luck adjustment beyond the runout: which of the hands it plays this way the bot happened to hold is luck too.
    The result is averaged over all of them, weighted by the probability that the bot takes its actions with each
    (the expectation given everything both players could see - the same mean, less variance)."""
    from headsup.lbr import COMBOS

    s = GameSession(opponent="call", advisor=None, device="cpu", seed=11)
    play_out(s)  # a calling station plays every hand the same way: its range at the showdown is every possible hand
    e = s.engine
    assert s.result["showdown"] and s.result["allin"] is None
    rng_info = s.result["vs_range"]
    want = _showdown_ev_against(np.ones(1326), list(e.hands[s.me]), list(e.board), min(e.bets))
    assert rng_info["ev"] == pytest.approx(want, abs=1e-6) and 0.0 <= rng_info["equity"] <= 1.0
    assert s.stats()["adj_total"] == pytest.approx(want, abs=1e-6) and s.hand_records[-1]["adj"] == pytest.approx(want, abs=1e-6)

    class PairsRaise:  # raises with a pocket pair, calls with everything else (never folds)
        game = s.game

        def probs(self, obs, ids=None):
            obs = np.asarray(obs)
            pair = obs[:, 0] == obs[:, 3]  # rank + 1 of the two hole cards
            out = np.zeros((len(obs), 4), dtype=np.float32)
            out[pair, 2], out[~pair, 1] = 1.0, 1.0
            return out

        def __call__(self, obs, ids=None):
            return self.probs(obs).argmax(1)

    s = GameSession(opponent="call", advisor=None, device="cpu", seed=4)
    s.opponent = PairsRaise()
    for _ in range(40):  # until the bot shows down a hand it raised with
        while not s.hand_over:
            s.bot_step() if s.bot_to_act else s.act("call")
        if any(entry["seat"] == "bot" and "raise" in entry["text"].lower() for entry in s.log) and s.result["showdown"]:
            break
        s.next_hand()
    e = s.engine
    assert e.hands[s.opp][0] % 13 == e.hands[s.opp][1] % 13  # it did raise: it holds a pair
    pairs = (COMBOS[:, 0] % 13 == COMBOS[:, 1] % 13).astype(float)
    want = _showdown_ev_against(pairs, list(e.hands[s.me]), list(e.board), min(e.bets))
    assert s.result["vs_range"]["ev"] == pytest.approx(want, abs=1e-6)
    json.dumps(s.state())


def test_luck_adjusted_results_have_the_same_mean_and_less_variance():
    s = GameSession(opponent="random", advisor=None, device="cpu", seed=2)
    s.runout_samples = 300  # sampled runouts where many cards are to come: the test is about the mean, not about exactness
    for _ in range(400):
        play_out(s)
        s.next_hand()
    raw, adj = np.asarray(s.results, dtype=float), np.asarray(s.adj_results, dtype=float)
    assert len(raw) == len(adj) == 400 and adj.std() < 0.85 * raw.std()
    assert abs(adj.mean() - raw.mean()) < 4 * np.sqrt((raw - adj).var() / len(raw))  # paired: the difference is pure luck
    st = s.stats()
    assert st["adj_total"] == pytest.approx(adj.sum()) and st["luck"] == pytest.approx(raw.sum() - adj.sum())
