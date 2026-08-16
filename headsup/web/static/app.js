/* Heads-Up Poker — browser front-end for headsup.web.server */
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const BOT_DELAY = 650;          // ms before the bot's next decision appears (natural pace)
const FLASH_MS = 1500;

let state = null;
let advice = null;
let peek = false;
let showAdvice = true;
let auto = { on: false, timer: null };
let botTimer = null;
let seen = { you: new Set(), bot: new Set(), board: new Set() };
let lastHand = null;
let lastBotFlash = null;
let players = null;

// ---------------------------------------------------------------- api
async function api(path, body) {
  const opts = body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) };
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* ignore */ }
  if (!res.ok || (data && data.error)) throw new Error((data && data.error) || res.statusText);
  return data;
}

async function refresh(newState) {
  state = newState || await api("/api/state" + (peek ? "?reveal=1" : ""));
  if (newState && peek && !state.hand_over) state = await api("/api/state?reveal=1");
  advice = (!state.hand_over && showAdvice) ? await api("/api/advice").catch(() => null) : null;
  render();
  scheduleBot();
  scheduleAuto();
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 2600);
}

const fmt = (n, sign = false) => {
  const v = Math.round(n * 100) / 100;
  const s = Math.abs(v) % 1 === 0 ? String(Math.abs(v)) : Math.abs(v).toFixed(2);
  return (v < 0 ? "−" : (sign && v > 0 ? "+" : "")) + s;
};
const pct = (p) => (100 * p).toFixed(0) + "%";
const speed = () => Number($("#speed").value);

// ---------------------------------------------------------------- bot pacing / autoplay
function scheduleBot() {
  clearTimeout(botTimer);
  botTimer = null;
  if (!state || !state.bot_to_act) return;
  const delay = auto.on ? Math.max(120, speed()) : BOT_DELAY;
  botTimer = setTimeout(async () => {
    botTimer = null;
    try { await botStep(); } catch (e) { toast(e.message); }
  }, delay);
}

function scheduleAuto() {
  clearTimeout(auto.timer);
  auto.timer = null;
  if (!auto.on || !state || state.bot_to_act) return;
  const d = Math.max(120, speed());
  if (state.your_turn) {
    auto.timer = setTimeout(async () => {
      auto.timer = null;
      try { await agentStep(); } catch (e) { toast(e.message); stopAuto(); }
    }, d);
  } else if (state.hand_over) {
    auto.timer = setTimeout(async () => {
      auto.timer = null;
      try { await nextHand(); } catch (e) { toast(e.message); stopAuto(); }
    }, d + 600);
  }
}

function startAuto() {
  auto.on = true;
  $("#btn-auto").textContent = "⏸ Stop autoplay";
  $("#btn-auto").classList.add("on");
  scheduleBot();
  scheduleAuto();
}
function stopAuto() {
  auto.on = false;
  clearTimeout(auto.timer);
  auto.timer = null;
  $("#btn-auto").textContent = "▶ Let the advisor play";
  $("#btn-auto").classList.remove("on");
}

// ---------------------------------------------------------------- actions
let busy = false;
async function mutate(fn) {
  // one request at a time: double clicks / Enter + click / autoplay ticks never overlap
  if (busy) return;
  busy = true;
  try { await fn(); } finally { busy = false; }
}
async function userAct(action) {
  if (!state || !state.your_turn) return;
  const a = state.actions.find((x) => x.action === action);
  if (!a) return;  // e.g. F with nothing to call
  await mutate(async () => {
    flash("#you-flash", a.label);
    await refresh(await api("/api/action", { action }));
  });
}
async function nextHand() {
  if (!state || !state.hand_over) return;
  await mutate(async () => { await refresh(await api("/api/next", {})); });
}
async function agentStep() {
  await mutate(async () => {
    const s = await api("/api/agent_step", { deterministic: $("#auto-argmax").checked });
    const last = s.log.length ? s.log[s.log.length - 1] : null;
    if (last && last.seat === "you" && s.hand_number === (state && state.hand_number)) flash("#you-flash", "Advisor: " + last.text);
    await refresh(s);
  });
}
async function botStep() {
  await mutate(async () => { await refresh(await api("/api/bot_step", {})); });
}

function flash(sel, text) {
  const el = $(sel);
  el.textContent = text;
  el.classList.remove("hidden", "fade");
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.classList.add("fade"); setTimeout(() => el.classList.add("hidden"), 500); }, FLASH_MS);
}

// ---------------------------------------------------------------- rendering helpers
function cardEl(c) {
  const el = document.createElement("div");
  if (c.hidden) {
    el.className = "card back";
  } else {
    el.className = "card" + (c.red ? " red" : "");
    el.innerHTML = `<div class="corner">${c.rank}<small>${c.suit}</small></div><div class="pip">${c.suit}</div>` +
                   `<div class="corner br">${c.rank}<small>${c.suit}</small></div>`;
  }
  return el;
}
const cardKey = (c, i) => (c.hidden ? "back" + i : "c" + c.id);

function renderCards(container, cards, zone, opts = {}) {
  container.innerHTML = "";
  let n = 0;
  cards.forEach((c, i) => {
    const key = cardKey(c, i);
    const el = cardEl(c);
    if (!seen[zone].has(key)) {
      el.classList.add(opts.flip ? "flip" : "deal");
      el.style.animationDelay = (n++ * 110) + "ms";
      seen[zone].add(key);
    }
    if (opts.dim) el.classList.add("dim");
    container.appendChild(el);
  });
}

function betEl(container, amount) {
  container.classList.remove("pop");
  if (!amount) { container.innerHTML = ""; return; }
  const cls = amount >= 50 ? "black" : amount >= 20 ? "green" : amount >= 5 ? "blue" : "";
  const changed = container.dataset.amount !== String(amount);
  container.innerHTML = `<span class="chip ${cls}"></span><span>${amount}</span>`;
  container.dataset.amount = String(amount);
  if (changed) { void container.offsetWidth; container.classList.add("pop"); }
}

function bars(container, rows, opts = {}) {
  container.innerHTML = "";
  if (!rows) return;
  const best = rows.reduce((b, r) => (r.p > b.p ? r : b), rows[0]);
  rows.forEach((r) => {
    const div = document.createElement("div");
    div.className = "bar" + (r === best && opts.markBest ? " best" : "") + (opts.chosen === r.action ? " chosen" : "");
    div.innerHTML = `<span>${r.label}</span><div class="track"><div class="fill" style="width:${(100 * r.p).toFixed(1)}%"></div></div><b>${pct(r.p)}</b>`;
    container.appendChild(div);
  });
}

const ACTION_LABELS = { fold: "Fold", call: "Check/Call", raise: "Raise", allin: "All-in" };
const ORDER = ["fold", "call", "raise", "allin"];

function miniCards(cards) {
  return cards.map((c) => `<span class="mini${c.red ? " red" : ""}">${c.rank}${c.suit}</span>`).join("");
}

// ---------------------------------------------------------------- render
function render() {
  const s = state;
  if (!s) return;
  if (s.hand_number !== lastHand) {
    seen = { you: new Set(), bot: new Set(), board: new Set() };
    lastHand = s.hand_number;
    lastBotFlash = null;
    $("#bot-flash").classList.add("hidden");
    $("#you-flash").classList.add("hidden");
  }
  const showdown = s.hand_over && s.result && s.result.showdown;

  // top bar
  const st = s.stats;
  const net = $("#net"); net.textContent = st.hands ? fmt(st.total, true) : "–"; net.className = "value " + (st.total > 0 ? "pos" : st.total < 0 ? "neg" : "");
  const avgTop = $("#avg-top"); avgTop.textContent = st.hands ? fmt(st.avg, true) : "–"; avgTop.className = "value " + (st.avg > 0 ? "pos" : st.avg < 0 ? "neg" : "");
  $("#hands").textContent = st.hands;
  $("#subtitle").textContent = `vs ${s.bot.label} · blinds ${s.settings.small_blind}/${s.settings.big_blind} · stacks ${s.settings.stack_size} · hand #${s.hand_number}`;

  // seats
  $("#bot-name").textContent = s.bot.label;
  $("#bot-stack").textContent = s.bot.stack;
  $("#you-stack").textContent = s.you.stack;
  for (const [id, seat] of [["bot", s.bot], ["you", s.you]]) {
    const pos = $(`#${id}-pos`);
    pos.textContent = seat.position === "dealer" ? "D · SB" : "BB";
    pos.className = "pos" + (seat.position === "dealer" ? " dealer" : "");
    const status = $(`#${id}-status`);
    if (seat.folded) { status.textContent = "folded"; status.className = "status folded"; }
    else if (seat.stack === 0 && !s.hand_over) { status.textContent = "all-in"; status.className = "status allin"; }
    else status.className = "status hidden";
    $(`#seat-${id}`).classList.toggle("active", !s.hand_over && ((id === "you" && s.your_turn) || (id === "bot" && s.bot_to_act)));
  }
  $("#you-class").textContent = s.you.hand_class ? "· " + s.you.hand_class : "";
  renderCards($("#bot-cards"), s.bot.cards, "bot", { flip: showdown || peek, dim: s.bot.folded });
  renderCards($("#you-cards"), s.you.cards, "you", { dim: s.you.folded });
  betEl($("#bot-bet"), s.bot.bet);
  betEl($("#you-bet"), s.you.bet);

  // center
  $("#pot").textContent = "Pot " + s.pot;
  const board = $("#board");
  board.innerHTML = "";
  let n = 0;
  for (let i = 0; i < 5; i++) {
    if (i < s.board.length) {
      const c = s.board[i], key = cardKey(c, i);
      const el = cardEl(c);
      if (!seen.board.has(key)) { el.classList.add("deal"); el.style.animationDelay = (n++ * 110) + "ms"; seen.board.add(key); }
      board.appendChild(el);
    } else {
      const slot = document.createElement("div"); slot.className = "slot"; board.appendChild(slot);
    }
  }
  $("#stage").textContent = s.stage.toUpperCase();

  // bot flash
  if (s.bot.last && s.bot.last.hand === s.hand_number) {
    const key = s.bot.last.hand + ":" + s.bot.last.n;
    if (key !== lastBotFlash) { lastBotFlash = key; flash("#bot-flash", "Bot " + s.bot.last.text); }
  }

  // actions
  const acts = $("#actions");
  acts.innerHTML = "";
  const best = advice && advice.probs ? advice.probs.reduce((b, r) => (r.p > b.p ? r : b), advice.probs[0]).action : null;
  s.actions.forEach((a) => {
    const b = document.createElement("button");
    b.className = "act " + a.action + (showAdvice && best === a.action ? " suggested" : "");
    b.innerHTML = `${a.label}<kbd>${a.key}</kbd>`;
    b.disabled = !s.your_turn;
    b.onclick = (ev) => { ev.currentTarget.blur(); userAct(a.action).catch((e) => toast(e.message)); };
    acts.appendChild(b);
  });
  if (!s.actions.length && !s.hand_over) {
    ORDER.forEach((a) => { const b = document.createElement("button"); b.className = "act " + a; b.disabled = true; b.textContent = ACTION_LABELS[a]; acts.appendChild(b); });
  }
  acts.classList.toggle("hidden", s.hand_over);
  $("#turn-hint").textContent = s.hand_over ? "" : s.your_turn ? (s.to_call ? `Your turn — ${s.to_call} to call` : "Your turn — check or bet") : "Bot is thinking…";

  // banner
  const banner = $("#banner");
  if (s.hand_over && s.result) {
    const r = s.result;
    const main = $("#banner-main");
    main.textContent = r.main;
    main.className = "banner-main " + (r.reward > 0 ? "win" : r.reward < 0 ? "lose" : "tie");
    $("#banner-sub").textContent = `${r.detail} · pot ${r.pot}`;
    banner.classList.remove("hidden");
  } else banner.classList.add("hidden");

  // bot panel
  const bl = s.bot.last;
  $("#bot-last").textContent = bl && bl.hand === s.hand_number ? bl.text : "–";
  if (bl && bl.probs) {
    bars($("#bot-bars"), ORDER.map((a, i) => ({ action: a, label: ACTION_LABELS[a], p: bl.probs[i] })), { chosen: bl.action });
    $("#bot-note").textContent = "action distribution of the bot at its last decision";
  } else {
    $("#bot-bars").innerHTML = "";
    $("#bot-note").textContent = bl ? "this bot has no distribution to show" : "";
  }

  // advisor panel
  $("#advisor-name").textContent = s.advisor.loaded ? "Advisor: " + s.advisor.name.split("/").pop() : "Advisor";
  if (advice && advice.equity) {
    const eq = advice.equity;
    $("#equity-text").textContent = `${pct(eq.win)} win · ${pct(eq.tie)} tie`;
    $("#eq-win").style.width = (100 * eq.win) + "%";
    $("#eq-tie").style.width = (100 * eq.tie) + "%";
  } else {
    $("#equity-text").textContent = "–"; $("#eq-win").style.width = "0"; $("#eq-tie").style.width = "0";
  }
  if (advice && advice.probs) {
    const b = advice.probs.reduce((x, r) => (r.p > x.p ? r : x), advice.probs[0]);
    $("#advice-best").textContent = `${b.label} (${pct(b.p)})`;
    bars($("#advice-bars"), advice.probs, { markBest: true });
    $("#advice-note").textContent = "what the DeepCFR policy would do in your spot";
  } else {
    $("#advice-best").textContent = "–";
    $("#advice-bars").innerHTML = "";
    $("#advice-note").textContent = !s.advisor.loaded ? (s.advisor.error || "no advisor policy loaded") : s.hand_over ? "" : "waiting for your turn";
  }

  // session
  drawSpark(st.cumulative);
  $("#wlt").textContent = `${st.won} / ${st.lost} / ${st.tied}`;
  $("#avg").textContent = st.hands ? `${fmt(st.avg, true)} chips/hand` : "–";
  if (st.hands) {
    const se = st.hands > 1 ? st.se_mbb : null;
    $("#mbb").textContent = `${fmt(Math.round(st.mbb), true).replace(/\B(?=(\d{3})+(?!\d))/g, "\u2009")}` + (se ? ` ± ${Math.round(se).toLocaleString("en").replace(/,/g, "\u2009")}` : "");
    $("#rate-note").textContent = st.hands < 50 ? `1 chip = ${1000 / s.settings.big_blind} mbb — rates are noisy below ~50 hands (${st.hands} so far)` : `${st.hands} hands`;
  } else { $("#mbb").textContent = "–"; $("#rate-note").textContent = "1 chip = " + (1000 / s.settings.big_blind) + " mbb"; }

  renderLog(s);
  renderHistory(s);
}

function renderLog(s) {
  const el = $("#log");
  el.innerHTML = "";
  let street = "preflop";
  el.insertAdjacentHTML("beforeend", `<div class="street">preflop</div>`);
  s.log.forEach((l) => {
    if (l.action === "board") {
      street = l.stage;
      const n = { flop: 3, turn: 4, river: 5 }[street] || 5;
      const from = { flop: 0, turn: 3, river: 4 }[street] || 0;
      const cards = s.board.slice(from, n);
      el.insertAdjacentHTML("beforeend", `<div class="street">${street} <span class="board-line">${miniCards(cards)}</span></div>`);
      return;
    }
    el.insertAdjacentHTML("beforeend", `<div class="line ${l.seat}"><span class="who">${l.seat === "you" ? "You" : "Bot"}</span><span>${l.text}</span></div>`);
  });
  if (s.hand_over && s.result) el.insertAdjacentHTML("beforeend", `<div class="street">${s.result.showdown ? "showdown" : "end"} <span class="board-line">${s.result.text}</span></div>`);
  el.scrollTop = el.scrollHeight;
}

function renderHistory(s) {
  const el = $("#history");
  el.innerHTML = "";
  s.history.forEach((h) => {
    const cls = h.reward > 0 ? "pos" : h.reward < 0 ? "neg" : "";
    const bot = h.bot ? ` vs ${miniCards(h.bot)}` : "";
    el.insertAdjacentHTML("beforeend",
      `<div class="h" title="${h.summary}"><span class="n">#${h.hand}</span><span>${miniCards(h.you)}${bot}` +
      `<div class="d">${h.position === "dealer" ? "D" : "BB"} · ${h.board.length ? miniCards(h.board) : "no flop"}</div></span>` +
      `<span class="r ${cls}">${fmt(h.reward, true)}</span></div>`);
  });
}

function drawSpark(hist) {
  const c = $("#spark"), ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 6;
  ctx.clearRect(0, 0, W, H);
  const data = [0].concat(hist || []);
  if (data.length < 2) { ctx.fillStyle = "#98a2ad"; ctx.font = "12px sans-serif"; ctx.fillText("play a few hands…", 8, H / 2 + 4); return; }
  const lo = Math.min(0, ...data), hi = Math.max(0, ...data), span = (hi - lo) || 1;
  const x = (i) => pad + (W - 2 * pad) * i / (data.length - 1);
  const y = (v) => H - pad - (H - 2 * pad) * (v - lo) / span;
  ctx.strokeStyle = "rgba(255,255,255,.15)"; ctx.beginPath(); ctx.moveTo(pad, y(0)); ctx.lineTo(W - pad, y(0)); ctx.stroke();
  const last = data[data.length - 1];
  ctx.strokeStyle = last >= 0 ? "#48c774" : "#ef5b5b"; ctx.lineWidth = 2; ctx.beginPath();
  data.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();
  ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(x(data.length - 1), y(last), 3, 0, 7); ctx.fill();
}

// ---------------------------------------------------------------- settings
async function openSettings() {
  try {
    players = players || (await api("/api/players")).players;
  } catch (e) { toast(e.message); return; }
  const set = state.settings;
  const fill = (sel, current, allowNone) => {
    sel.innerHTML = "";
    if (allowNone) { const o = document.createElement("option"); o.value = ""; o.textContent = "none"; sel.appendChild(o); }
    let found = false;
    players.forEach((p) => { const o = document.createElement("option"); o.value = p.spec; o.textContent = p.label; if (p.spec === current) { o.selected = true; found = true; } sel.appendChild(o); });
    if (current && !found) { const o = document.createElement("option"); o.value = current; o.textContent = current; o.selected = true; sel.appendChild(o); }
  };
  fill($("#opponent-select"), set.opponent, false);
  fill($("#advisor-select"), set.advisor || "", true);
  const f = $("#settings-form");
  f.stack_size.value = set.stack_size; f.small_blind.value = set.small_blind; f.big_blind.value = set.big_blind;
  f.raise_cap.value = set.raise_cap; f.seed.value = set.seed == null ? "" : set.seed; f.deterministic.checked = !!set.deterministic;
  f.opponent_custom.value = "";
  $("#modal").classList.remove("hidden");
}

$("#settings-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    opponent: f.opponent_custom.value.trim() || f.opponent.value,
    advisor: f.advisor.value,
    stack_size: Number(f.stack_size.value), small_blind: Number(f.small_blind.value), big_blind: Number(f.big_blind.value),
    raise_cap: Number(f.raise_cap.value), seed: f.seed.value.trim(), deterministic: f.deterministic.checked,
  };
  try {
    stopAuto();
    const s = await api("/api/new", body);
    $("#modal").classList.add("hidden");
    lastHand = null;
    await refresh(s);
    toast("New table started");
  } catch (err) { toast(err.message); }
};

// ---------------------------------------------------------------- wiring
$("#btn-settings").onclick = openSettings;
$("#btn-cancel").onclick = () => $("#modal").classList.add("hidden");
$("#modal").onclick = (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); };
$("#btn-next").onclick = (e) => { e.currentTarget.blur(); nextHand().catch((err) => toast(err.message)); };
$("#btn-auto").onclick = () => (auto.on ? stopAuto() : startAuto());
$("#btn-step").onclick = async () => { stopAuto(); try { if (state.your_turn) await agentStep(); else if (state.hand_over) await nextHand(); } catch (e) { toast(e.message); } };
$("#toggle-peek").onchange = async (e) => { peek = e.target.checked; await refresh().catch((err) => toast(err.message)); };
$("#toggle-advice").onchange = async (e) => { showAdvice = e.target.checked; $("#advice-body").classList.toggle("hidden", !showAdvice); await refresh().catch((err) => toast(err.message)); };
$("#speed").oninput = () => { if (auto.on) { scheduleBot(); scheduleAuto(); } };

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (!$("#modal").classList.contains("hidden")) { if (e.key === "Escape") $("#modal").classList.add("hidden"); return; }
  const k = e.key.toLowerCase();
  const map = { f: "fold", c: "call", r: "raise", a: "allin" };
  if (k === "enter" || k === " ") e.preventDefault();  // never let a focused button double-fire
  if (map[k] && state && state.your_turn) { e.preventDefault(); userAct(map[k]).catch((err) => toast(err.message)); }
  else if ((k === "enter" || k === " " || k === "n") && state && state.hand_over) { nextHand().catch((err) => toast(err.message)); }
  else if (k === "s") { $("#toggle-peek").checked = !peek; $("#toggle-peek").dispatchEvent(new Event("change")); }
  else if (k === "d") { $("#toggle-advice").checked = !showAdvice; $("#toggle-advice").dispatchEvent(new Event("change")); }
  else if (k === "p") { auto.on ? stopAuto() : startAuto(); }
});

refresh().then(() => { if (location.hash === "#settings") openSettings(); }).catch((e) => toast(e.message));
