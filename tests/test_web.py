import http.client
import json
import threading
from http.server import ThreadingHTTPServer

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
    assert [a["action"] for a in st["actions"]] == ["fold", "call", "raise", "allin"]
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
    assert labels[1] == "Check" and labels[2] == "Bet 2"


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
