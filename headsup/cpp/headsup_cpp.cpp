// C++ kernels for headsup-poker: game engine, treys-compatible hand evaluator,
// the DeepCFR network forward pass, external-sampling MCCFR traversals and an
// envpool-style vectorised environment.  Exposed to Python via pybind11 as `headsup_cpp`.
//
// Everything mirrors the Python reference implementation in headsup/engine.py,
// headsup/numpy_model.py and headsup/deepcfr/traverse.py bit-for-bit (tests compare them).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace hp {

constexpr int OBS_DIM = 31;
constexpr int NUM_ACTIONS = 4;
constexpr int NUM_CARDS = 52;
enum Action { FOLD = 0, CHECK_CALL = 1, RAISE = 2, ALL_IN = 3 };
enum Stage { PREFLOP = 0, FLOP = 1, TURN = 2, RIVER = 3, END = 4 };
constexpr int BOARD_CARDS_BY_STAGE[5] = {0, 3, 4, 5, 5};

// ------------------------------------------------------------------ hand evaluator
// treys card encoding: bitrank(16+) | suit(12..15) | rank(8..11) | prime(0..7)
constexpr uint32_t PRIMES[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

inline uint32_t treys_card(int c) {
  const int r = c % 13, s = c / 13;
  return (1u << r << 16) | (uint32_t(1u << s) << 12) | (uint32_t(r) << 8) | PRIMES[r];
}

struct Tables {
  std::unordered_map<uint32_t, uint16_t> flush, unsuited;
  bool ready = false;
} g_tables;

inline uint32_t prime_product_from_rankbits(uint32_t bits) {
  uint32_t p = 1;
  for (int i = 0; i < 13; ++i)
    if (bits & (1u << i)) p *= PRIMES[i];
  return p;
}

inline int eval5(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t c4) {
  if (c0 & c1 & c2 & c3 & c4 & 0xF000u) {
    const uint32_t bits = (c0 | c1 | c2 | c3 | c4) >> 16;
    return g_tables.flush.at(prime_product_from_rankbits(bits));
  }
  const uint32_t prod = (c0 & 0xFF) * (c1 & 0xFF) * (c2 & 0xFF) * (c3 & 0xFF) * (c4 & 0xFF);
  return g_tables.unsuited.at(prod);
}

// best 5 of 7 (lower is better), same as treys Evaluator._seven
int eval7(const int* cards7) {
  uint32_t c[7];
  for (int i = 0; i < 7; ++i) c[i] = treys_card(cards7[i]);
  int best = 7462;
  for (int a = 0; a < 3; ++a)
    for (int b = a + 1; b < 4; ++b)
      for (int d = b + 1; d < 5; ++d)
        for (int e = d + 1; e < 6; ++e)
          for (int f = e + 1; f < 7; ++f) best = std::min(best, eval5(c[a], c[b], c[d], c[e], c[f]));
  return best;
}

// ------------------------------------------------------------------ engine
struct EngineConfig {
  int stack_size = 100;
  int small_blind = 1;
  int big_blind = 2;
  int raise_cap = 3;
};

struct Engine {
  EngineConfig cfg;
  int hands[2][2] = {{0, 1}, {2, 3}};
  int board[5] = {4, 5, 6, 7, 8};
  int stacks[2] = {0, 0}, bets[2] = {0, 0}, stage_bets[2] = {0, 0};
  int pot = 0, stage = PREFLOP, current = 0, folded = -1, acted = 0, consecutive_raises = 0;
  int rewards[2] = {0, 0};
  bool done = true;
  static constexpr int dealer = 0;

  explicit Engine(EngineConfig c = EngineConfig()) : cfg(c) {}

  void reset(const int* deck9) {
    hands[0][0] = deck9[0];
    hands[0][1] = deck9[1];
    hands[1][0] = deck9[2];
    hands[1][1] = deck9[3];
    for (int i = 0; i < 5; ++i) board[i] = deck9[4 + i];
    stacks[0] = cfg.stack_size - cfg.small_blind;
    stacks[1] = cfg.stack_size - cfg.big_blind;
    bets[0] = stage_bets[0] = cfg.small_blind;
    bets[1] = stage_bets[1] = cfg.big_blind;
    pot = cfg.small_blind + cfg.big_blind;
    stage = PREFLOP;
    current = dealer;
    folded = -1;
    acted = 0;
    consecutive_raises = 0;
    done = false;
    rewards[0] = rewards[1] = 0;
  }

  template <class RNG>
  void reset_random(RNG& rng) {
    // partial Fisher-Yates: 9 distinct cards out of 52
    int deck[NUM_CARDS];
    for (int i = 0; i < NUM_CARDS; ++i) deck[i] = i;
    int out[9];
    for (int i = 0; i < 9; ++i) {
      std::uniform_int_distribution<int> d(i, NUM_CARDS - 1);
      const int j = d(rng);
      std::swap(deck[i], deck[j]);
      out[i] = deck[i];
    }
    reset(out);
  }

  void observation(int seat, float* obs) const {
    const int p = seat < 0 ? current : seat;
    const int o = 1 - p;
    std::memset(obs, 0, sizeof(float) * OBS_DIM);
    auto put = [&](int slot, int card) {
      obs[3 * slot + 0] = float(card % 13 + 1);
      obs[3 * slot + 1] = float(card / 13 + 1);
      obs[3 * slot + 2] = float(card + 1);
    };
    put(0, hands[p][0]);
    put(1, hands[p][1]);
    const int nb = BOARD_CARDS_BY_STAGE[stage];
    for (int i = 0; i < nb; ++i) put(2 + i, board[i]);
    obs[21] = float(stage);
    obs[22] = float(p != dealer);
    const float fpot = float(pot);
    const int stack = stacks[p];
    const int diff = stage_bets[o] - stage_bets[p];
    obs[23] = float(diff) / fpot;
    obs[24] = float(bets[p]) / fpot;
    obs[25] = float(bets[o]) / fpot;
    obs[26] = float(stage_bets[p]) / fpot;
    obs[27] = float(stage_bets[o]) / fpot;
    obs[28] = float(stack) / fpot;
    obs[29] = fpot / 1000.0f;
    obs[30] = stack > 0 ? float(diff) / float(stack) : 0.0f;
  }

  bool street_finished() const {
    if (acted != 3) return false;
    if (stage_bets[0] == stage_bets[1]) return true;
    const int s = stage_bets[0] < stage_bets[1] ? 0 : 1;
    return stacks[s] == 0;
  }

  void showdown() {
    stage = END;
    int c0[7] = {hands[0][0], hands[0][1], board[0], board[1], board[2], board[3], board[4]};
    int c1[7] = {hands[1][0], hands[1][1], board[0], board[1], board[2], board[3], board[4]};
    const int s0 = eval7(c0), s1 = eval7(c1);
    const int won = std::min(bets[0], bets[1]);
    if (s0 == s1) {
      rewards[0] = rewards[1] = 0;
    } else if (s0 < s1) {
      rewards[0] = won;
      rewards[1] = -won;
    } else {
      rewards[0] = -won;
      rewards[1] = won;
    }
    done = true;
  }

  // returns true when the hand is over
  bool fold_allowed() const { return stage_bets[1 - current] - stage_bets[current] > 0; }

  bool step(int action) {
    const int p = current, o = 1 - p;
    if (action == FOLD && stage_bets[o] == stage_bets[p]) action = CHECK_CALL;  // nothing to call: fold is a dominated check
    if (action == RAISE) {
      if (++consecutive_raises >= cfg.raise_cap) action = ALL_IN;
    } else {
      consecutive_raises = 0;
    }
    if (action == FOLD) {
      folded = p;
      rewards[o] = bets[p];
      rewards[p] = -bets[p];
      done = true;
      return true;
    }
    int amount;
    if (action == CHECK_CALL)
      amount = std::min(stage_bets[o] - stage_bets[p], stacks[p]);
    else if (action == RAISE)
      amount = std::min(stage_bets[o] - stage_bets[p] + cfg.big_blind, stacks[p]);
    else
      amount = stacks[p];
    bets[p] += amount;
    stage_bets[p] += amount;
    stacks[p] -= amount;
    pot += amount;
    acted |= 1 << p;
    current = o;
    if (street_finished()) {
      if (stage == RIVER || std::min(stacks[0], stacks[1]) == 0) {
        showdown();
        return true;
      }
      stage += 1;
      stage_bets[0] = stage_bets[1] = 0;
      acted = 0;
      consecutive_raises = 0;
      current = 1 - dealer;
    }
    return false;
  }
};

// ------------------------------------------------------------------ network
struct Linear {
  int in = 0, out = 0;
  std::vector<float> w, b;  // w is (out, in) row-major
  void apply(const float* x, float* y) const {
    for (int o = 0; o < out; ++o) {
      const float* row = w.data() + size_t(o) * in;
      float acc = b[o];
      for (int i = 0; i < in; ++i) acc += row[i] * x[i];
      y[o] = acc;
    }
  }
};

inline void relu(float* x, int n) {
  for (int i = 0; i < n; ++i) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

struct Model {
  static constexpr int D = 64;
  std::vector<float> rank_emb, suit_emb, card_emb, stage_emb, first_emb;
  Linear card_fc[3], stage_fc[2], bets_fc[2], comb[3], head;

  void forward(const float* obs, float* out) const {
    float emb[7][D];
    for (int s = 0; s < 7; ++s) {
      const int r = int(obs[3 * s]), su = int(obs[3 * s + 1]), c = int(obs[3 * s + 2]);
      const float* er = rank_emb.data() + size_t(r) * D;
      const float* es = suit_emb.data() + size_t(su) * D;
      const float* ec = card_emb.data() + size_t(c) * D;
      for (int i = 0; i < D; ++i) emb[s][i] = er[i] + es[i] + ec[i];
    }
    float x[4 * D];
    for (int i = 0; i < D; ++i) {
      x[i] = emb[0][i] + emb[1][i];
      x[D + i] = emb[2][i] + emb[3][i] + emb[4][i];
      x[2 * D + i] = emb[5][i];
      x[3 * D + i] = emb[6][i];
    }
    float c1[D], c2[D];
    card_fc[0].apply(x, c1);
    relu(c1, D);
    card_fc[1].apply(c1, c2);
    relu(c2, D);
    card_fc[2].apply(c2, c1);
    relu(c1, D);  // c1 = card features

    float se[2 * D];
    const int st = int(obs[21]), fa = int(obs[22]);
    std::memcpy(se, stage_emb.data() + size_t(st) * D, sizeof(float) * D);
    std::memcpy(se + D, first_emb.data() + size_t(fa) * D, sizeof(float) * D);
    float s1[D], s2[D];
    stage_fc[0].apply(se, s1);
    relu(s1, D);
    stage_fc[1].apply(s1, s2);
    relu(s2, D);  // s2 = stage features

    float b1[D], b2[D];
    bets_fc[0].apply(obs + 23, b1);
    relu(b1, D);
    bets_fc[1].apply(b1, b2);
    for (int i = 0; i < D; ++i) b2[i] += b1[i];
    relu(b2, D);  // b2 = bet features

    float z[3 * D];
    std::memcpy(z, c1, sizeof(float) * D);
    std::memcpy(z + D, s2, sizeof(float) * D);
    std::memcpy(z + 2 * D, b2, sizeof(float) * D);
    float z1[D], z2[D];
    comb[0].apply(z, z1);
    relu(z1, D);
    comb[1].apply(z1, z2);
    for (int i = 0; i < D; ++i) z2[i] += z1[i];
    relu(z2, D);
    comb[2].apply(z2, z1);
    for (int i = 0; i < D; ++i) z1[i] += z2[i];
    relu(z1, D);
    // normalize: (z - mean) / (std_unbiased + 1e-6)
    float mean = 0.0f;
    for (int i = 0; i < D; ++i) mean += z1[i];
    mean /= D;
    float var = 0.0f;
    for (int i = 0; i < D; ++i) {
      const float d = z1[i] - mean;
      var += d * d;
    }
    const float sd = std::sqrt(var / (D - 1));
    for (int i = 0; i < D; ++i) z1[i] = (z1[i] - mean) / (sd + 1e-6f);
    head.apply(z1, out);
  }
};

std::vector<float> to_vec(const py::dict& d, const char* key, size_t expect) {
  if (!d.contains(key)) throw std::runtime_error(std::string("missing weight ") + key);
  auto arr = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(d[key]);
  if (!arr) throw std::runtime_error(std::string("bad weight ") + key);
  if (size_t(arr.size()) != expect)
    throw std::runtime_error(std::string("unexpected size for ") + key + ": " + std::to_string(arr.size()) +
                             " != " + std::to_string(expect));
  std::vector<float> v(arr.size());
  std::memcpy(v.data(), arr.data(), sizeof(float) * arr.size());
  return v;
}

Linear to_linear(const py::dict& d, const std::string& prefix, int in, int out) {
  Linear l;
  l.in = in;
  l.out = out;
  l.w = to_vec(d, (prefix + ".weight").c_str(), size_t(in) * out);
  l.b = to_vec(d, (prefix + ".bias").c_str(), out);
  return l;
}

std::shared_ptr<Model> model_from_dict(const py::dict& d) {
  auto m = std::make_shared<Model>();
  const int D = Model::D;
  m->rank_emb = to_vec(d, "card_model.cards_embeddings.rank_embedding.weight", 14 * D);
  m->suit_emb = to_vec(d, "card_model.cards_embeddings.suit_embedding.weight", 5 * D);
  m->card_emb = to_vec(d, "card_model.cards_embeddings.card_embedding.weight", 53 * D);
  m->card_fc[0] = to_linear(d, "card_model.fc1", 4 * D, D);
  m->card_fc[1] = to_linear(d, "card_model.fc2", D, D);
  m->card_fc[2] = to_linear(d, "card_model.fc3", D, D);
  m->stage_emb = to_vec(d, "stage_and_order_model.stage_embedding.weight", 4 * D);
  m->first_emb = to_vec(d, "stage_and_order_model.first_to_act_embedding.weight", 2 * D);
  m->stage_fc[0] = to_linear(d, "stage_and_order_model.fc1", 2 * D, D);
  m->stage_fc[1] = to_linear(d, "stage_and_order_model.fc2", D, D);
  m->bets_fc[0] = to_linear(d, "bets_model.fc1", 8, D);
  m->bets_fc[1] = to_linear(d, "bets_model.fc2", D, D);
  m->comb[0] = to_linear(d, "comb1", 3 * D, D);
  m->comb[1] = to_linear(d, "comb2", D, D);
  m->comb[2] = to_linear(d, "comb3", D, D);
  m->head = to_linear(d, "action_head", D, NUM_ACTIONS);
  return m;
}

inline void regret_matching(const float* adv, float* sigma, bool fold_allowed = true) {
  float total = 0.0f;
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    sigma[a] = (adv[a] > 0.0f && (a != FOLD || fold_allowed)) ? adv[a] : 0.0f;
    total += sigma[a];
  }
  if (total <= 1e-6f) {
    const int n = fold_allowed ? NUM_ACTIONS : NUM_ACTIONS - 1;
    for (int a = 0; a < NUM_ACTIONS; ++a) sigma[a] = (a == FOLD && !fold_allowed) ? 0.0f : 1.0f / n;
  } else {
    for (int a = 0; a < NUM_ACTIONS; ++a) sigma[a] /= total;
  }
}

inline void softmax(const float* logits, float* p) {
  float m = logits[0];
  for (int a = 1; a < NUM_ACTIONS; ++a) m = std::max(m, logits[a]);
  float s = 0.0f;
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    p[a] = std::exp(logits[a] - m);
    s += p[a];
  }
  for (int a = 0; a < NUM_ACTIONS; ++a) p[a] /= s;
}

template <class RNG>
inline int sample(const float* p, RNG& rng) {
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  const float u = u01(rng);
  float acc = 0.0f;
  for (int a = 0; a < NUM_ACTIONS - 1; ++a) {
    acc += p[a];
    if (u < acc) return a;
  }
  return NUM_ACTIONS - 1;
}

// ------------------------------------------------------------------ MCCFR traversal
struct Memory {
  std::vector<float> obs, t, target;
  void add(const float* o, float tt, const float* tg) {
    obs.insert(obs.end(), o, o + OBS_DIM);
    t.push_back(tt);
    target.insert(target.end(), tg, tg + NUM_ACTIONS);
  }
  size_t size() const { return t.size(); }
};

struct Traverser {
  const Model* nets[2];
  int traverser;
  float t;
  std::mt19937_64 rng;
  Memory adv, strat;
  long nodes = 0;

  float traverse(Engine& e) {
    if (e.done) return float(e.rewards[traverser]);
    const int p = e.current;
    float obs[OBS_DIM], values[NUM_ACTIONS], sigma[NUM_ACTIONS];
    e.observation(-1, obs);
    nets[p]->forward(obs, values);
    const bool fold_ok = e.fold_allowed();
    regret_matching(values, sigma, fold_ok);
    ++nodes;
    if (p == traverser) {
      float va[NUM_ACTIONS];
      for (int a = 0; a < NUM_ACTIONS; ++a) {
        if (a == FOLD && !fold_ok) continue;  // fold == check here; filled in below
        if (a + 1 < NUM_ACTIONS) {
          Engine child = e;
          child.step(a);
          va[a] = traverse(child);
        } else {
          e.step(a);
          va[a] = traverse(e);
        }
      }
      if (!fold_ok) va[FOLD] = va[CHECK_CALL];
      float mean = 0.0f;
      for (int a = 0; a < NUM_ACTIONS; ++a) mean += sigma[a] * va[a];
      for (int a = 0; a < NUM_ACTIONS; ++a) va[a] -= mean;
      adv.add(obs, t, va);
      return mean;
    }
    strat.add(obs, t, sigma);
    e.step(sample(sigma, rng));
    return traverse(e);
  }
};

py::array_t<float> to_array(std::vector<float>& v, ssize_t rows, ssize_t cols) {
  auto* holder = new std::vector<float>(std::move(v));
  py::capsule free_when_done(holder, [](void* p) { delete reinterpret_cast<std::vector<float>*>(p); });
  if (cols == 1)
    return py::array_t<float>({rows}, {sizeof(float)}, holder->data(), free_when_done);
  return py::array_t<float>({rows, cols}, {sizeof(float) * cols, sizeof(float)}, holder->data(), free_when_done);
}

py::tuple run_traversals(std::shared_ptr<Model> net0, std::shared_ptr<Model> net1, int traverser, int n_traversals,
                         float t, uint64_t seed, EngineConfig cfg, py::object decks_obj) {
  Traverser tr;
  tr.nets[0] = net0.get();
  tr.nets[1] = net1.get();
  tr.traverser = traverser;
  tr.t = t;
  tr.rng.seed(seed);
  std::vector<int> decks;  // optional fixed deals, (n_traversals, 9)
  if (!decks_obj.is_none()) {
    auto arr = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(decks_obj);
    if (!arr || arr.ndim() != 2 || arr.shape(0) != n_traversals || arr.shape(1) < 9)
      throw std::runtime_error("decks must be an int array of shape (n_traversals, >=9)");
    decks.assign(arr.data(), arr.data() + arr.size());
    const int stride = int(arr.shape(1));
    py::gil_scoped_release release;
    Engine e(cfg);
    for (int i = 0; i < n_traversals; ++i) {
      e.reset(decks.data() + size_t(i) * stride);
      tr.traverse(e);
    }
  } else {
    py::gil_scoped_release release;
    Engine e(cfg);
    for (int i = 0; i < n_traversals; ++i) {
      e.reset_random(tr.rng);
      tr.traverse(e);
    }
  }
  const ssize_t na = ssize_t(tr.adv.size()), ns = ssize_t(tr.strat.size());
  return py::make_tuple(to_array(tr.adv.obs, na, OBS_DIM), to_array(tr.adv.t, na, 1),
                        to_array(tr.adv.target, na, NUM_ACTIONS), to_array(tr.strat.obs, ns, OBS_DIM),
                        to_array(tr.strat.t, ns, 1), to_array(tr.strat.target, ns, NUM_ACTIONS), tr.nodes);
}

// ------------------------------------------------------------------ vectorised env
struct VecEnv {
  enum OpponentKind { RANDOM = 0, CALL = 1, ALLIN = 2, RAISE_ = 3, MODEL = 4 };
  std::vector<Engine> engines;
  std::vector<int> agent_seat;
  bool alternate;
  std::mt19937_64 rng;
  OpponentKind opp_kind = CALL;
  std::shared_ptr<Model> opp_model;
  bool opp_deterministic = false;
  std::vector<float> last_probs;  // (num_envs, 4) of the opponent's last decision per table
  long hands_completed = 0;

  VecEnv(int n, uint64_t seed, EngineConfig cfg, bool alternate_seats)
      : engines(size_t(n), Engine(cfg)), agent_seat(size_t(n)), alternate(alternate_seats), rng(seed),
        last_probs(size_t(n) * NUM_ACTIONS, 0.25f) {
    for (int i = 0; i < n; ++i) agent_seat[i] = (i % 2) ^ 1;  // flipped on first reset
  }

  int opponent_action(int i, const float* obs) {
    switch (opp_kind) {
      case RANDOM: {
        std::uniform_int_distribution<int> d(engines[i].fold_allowed() ? 0 : 1, NUM_ACTIONS - 1);
        return d(rng);
      }
      case CALL:
        return CHECK_CALL;
      case ALLIN:
        return ALL_IN;
      case RAISE_:
        return RAISE;
      case MODEL: {
        float logits[NUM_ACTIONS];
        float* p = last_probs.data() + size_t(i) * NUM_ACTIONS;
        opp_model->forward(obs, logits);
        if (!engines[i].fold_allowed()) logits[FOLD] = -1e30f;
        softmax(logits, p);
        if (opp_deterministic) return int(std::max_element(p, p + NUM_ACTIONS) - p);
        return sample(p, rng);
      }
    }
    return CHECK_CALL;
  }

  void reset_engine(int i) {
    if (alternate)
      agent_seat[i] ^= 1;
    else {
      std::uniform_int_distribution<int> d(0, 1);
      agent_seat[i] = d(rng);
    }
    engines[i].reset_random(rng);
  }

  void advance_opponent(int i) {
    Engine& e = engines[i];
    float obs[OBS_DIM];
    while (!e.done && e.current != agent_seat[i]) {
      e.observation(-1, obs);
      e.step(opponent_action(i, obs));
    }
  }

  void write_obs(py::array_t<float>& out) {
    auto o = out.mutable_unchecked<2>();
    for (size_t i = 0; i < engines.size(); ++i) engines[i].observation(agent_seat[i], o.mutable_data(i, 0));
  }

  py::array_t<float> reset() {
    for (size_t i = 0; i < engines.size(); ++i) {
      reset_engine(int(i));
      advance_opponent(int(i));
    }
    py::array_t<float> obs({ssize_t(engines.size()), ssize_t(OBS_DIM)});
    write_obs(obs);
    return obs;
  }

  py::tuple step(py::array_t<int64_t, py::array::c_style | py::array::forcecast> actions, bool auto_reset) {
    if (actions.size() != ssize_t(engines.size())) throw std::runtime_error("actions has wrong length");
    const ssize_t n = ssize_t(engines.size());
    py::array_t<float> rewards(std::vector<ssize_t>{n});
    py::array_t<bool> dones(std::vector<ssize_t>{n});
    auto r = rewards.mutable_unchecked<1>();
    auto d = dones.mutable_unchecked<1>();
    auto a = actions.unchecked<1>();
    for (ssize_t i = 0; i < n; ++i) {
      Engine& e = engines[i];
      r(i) = 0.0f;
      if (!e.done) {
        e.step(int(a(i)));
        advance_opponent(int(i));
      }
      d(i) = e.done;
      if (e.done) {
        r(i) = float(e.rewards[agent_seat[i]]);
        ++hands_completed;
        if (auto_reset) {
          reset_engine(int(i));
          advance_opponent(int(i));
        }
      }
    }
    py::array_t<float> obs({n, ssize_t(OBS_DIM)});
    write_obs(obs);
    return py::make_tuple(obs, rewards, dones);
  }
};


// ------------------------------------------------------------------ micro benchmarks (for profiling)
double bench_engine(int hands, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Engine e;
  std::uniform_int_distribution<int> d(0, 3);
  long steps = 0;
  for (int i = 0; i < hands; ++i) {
    e.reset_random(rng);
    while (!e.done) { e.step(d(rng)); ++steps; }
  }
  return double(steps);
}
double bench_eval7(int n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  long acc = 0;
  Engine e;
  for (int i = 0; i < n; ++i) {
    e.reset_random(rng);
    int c[7] = {e.hands[0][0], e.hands[0][1], e.board[0], e.board[1], e.board[2], e.board[3], e.board[4]};
    acc += eval7(c);
  }
  return double(acc);
}
double bench_forward(const Model& m, int n) {
  float obs[OBS_DIM] = {0}, out[NUM_ACTIONS];
  Engine e; std::mt19937_64 rng(1); e.reset_random(rng); e.observation(-1, obs);
  float acc = 0;
  for (int i = 0; i < n; ++i) { m.forward(obs, out); acc += out[0]; }
  return double(acc);
}

}  // namespace hp

PYBIND11_MODULE(headsup_cpp, m) {
  using namespace hp;
  m.doc() = "C++ kernels for headsup-poker";

  m.def(
      "set_tables",
      [](py::array_t<uint32_t> fk, py::array_t<uint16_t> fv, py::array_t<uint32_t> uk, py::array_t<uint16_t> uv) {
        g_tables.flush.clear();
        g_tables.unsuited.clear();
        auto a = fk.unchecked<1>(), c = uk.unchecked<1>();
        auto b = fv.unchecked<1>(), d = uv.unchecked<1>();
        for (ssize_t i = 0; i < a.shape(0); ++i) g_tables.flush[a(i)] = b(i);
        for (ssize_t i = 0; i < c.shape(0); ++i) g_tables.unsuited[c(i)] = d(i);
        g_tables.ready = true;
      },
      "Install treys lookup tables (prime product -> rank).");
  m.def("tables_ready", []() { return g_tables.ready; });
  m.def("bench_engine", &bench_engine);
  m.def("bench_eval7", &bench_eval7);
  m.def("bench_forward", [](const Model& mm, int n) { return bench_forward(mm, n); });
  m.def("eval7", [](std::vector<int> cards) {
    if (cards.size() != 7) throw std::runtime_error("need 7 cards");
    return eval7(cards.data());
  });

  py::class_<EngineConfig>(m, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("stack_size", &EngineConfig::stack_size)
      .def_readwrite("small_blind", &EngineConfig::small_blind)
      .def_readwrite("big_blind", &EngineConfig::big_blind)
      .def_readwrite("raise_cap", &EngineConfig::raise_cap);

  py::class_<Engine>(m, "Engine")
      .def(py::init<EngineConfig>(), py::arg("cfg") = EngineConfig())
      .def("reset",
           [](Engine& e, std::vector<int> deck) {
             if (deck.size() < 9) throw std::runtime_error("deck needs >= 9 cards");
             e.reset(deck.data());
           })
      .def("step", [](Engine& e, int a) { return e.step(a); })
      .def("observation",
           [](const Engine& e, int seat) {
             py::array_t<float> out(std::vector<ssize_t>{OBS_DIM});
             e.observation(seat, out.mutable_data());
             return out;
           },
           py::arg("seat") = -1)
      .def_property_readonly("done", [](const Engine& e) { return e.done; })
      .def_property_readonly("current", [](const Engine& e) { return e.current; })
      .def_property_readonly("stage", [](const Engine& e) { return e.stage; })
      .def_property_readonly("pot", [](const Engine& e) { return e.pot; })
      .def_property_readonly("folded", [](const Engine& e) { return e.folded; })
      .def_property_readonly("rewards", [](const Engine& e) { return std::vector<int>{e.rewards[0], e.rewards[1]}; })
      .def_property_readonly("stacks", [](const Engine& e) { return std::vector<int>{e.stacks[0], e.stacks[1]}; })
      .def_property_readonly("bets", [](const Engine& e) { return std::vector<int>{e.bets[0], e.bets[1]}; });

  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init([](py::dict d) { return model_from_dict(d); }))
      .def("forward", [](const Model& mm, py::array_t<float, py::array::c_style | py::array::forcecast> obs) {
        if (obs.ndim() == 1) {
          py::array_t<float> out(std::vector<ssize_t>{NUM_ACTIONS});
          mm.forward(obs.data(), out.mutable_data());
          return out;
        }
        const ssize_t n = obs.shape(0);
        py::array_t<float> out({n, ssize_t(NUM_ACTIONS)});
        auto o = out.mutable_unchecked<2>();
        auto in = obs.unchecked<2>();
        for (ssize_t i = 0; i < n; ++i) mm.forward(in.data(i, 0), o.mutable_data(i, 0));
        return out;
      });

  m.def("run_traversals", &run_traversals, py::arg("net0"), py::arg("net1"), py::arg("traverser"), py::arg("n_traversals"),
        py::arg("t"), py::arg("seed"), py::arg("cfg") = EngineConfig(), py::arg("decks") = py::none(),
        "External-sampling MCCFR traversals; returns (adv_obs, adv_t, adv_target, strat_obs, strat_t, strat_target, nodes)");

  py::class_<VecEnv>(m, "VecEnv")
      .def(py::init<int, uint64_t, EngineConfig, bool>(), py::arg("num_envs"), py::arg("seed"),
           py::arg("cfg") = EngineConfig(), py::arg("alternate_seats") = true)
      .def("set_opponent_simple",
           [](VecEnv& v, const std::string& kind) {
             if (kind == "random")
               v.opp_kind = VecEnv::RANDOM;
             else if (kind == "call")
               v.opp_kind = VecEnv::CALL;
             else if (kind == "allin")
               v.opp_kind = VecEnv::ALLIN;
             else if (kind == "raise")
               v.opp_kind = VecEnv::RAISE_;
             else
               throw std::runtime_error("unknown simple opponent " + kind);
           })
      .def("set_opponent_model",
           [](VecEnv& v, std::shared_ptr<Model> mdl, bool deterministic) {
             v.opp_model = std::move(mdl);
             v.opp_kind = VecEnv::MODEL;
             v.opp_deterministic = deterministic;
           },
           py::arg("model"), py::arg("deterministic") = false)
      .def("reset", &VecEnv::reset)
      .def("step", &VecEnv::step, py::arg("actions"), py::arg("auto_reset") = true)
      .def_property_readonly("num_envs", [](const VecEnv& v) { return int(v.engines.size()); })
      .def_property_readonly("agent_seat", [](const VecEnv& v) { return v.agent_seat; })
      .def_property_readonly("hands_completed", [](const VecEnv& v) { return v.hands_completed; })
      .def_property_readonly("last_probs",
                             [](const VecEnv& v) {
                               py::array_t<float> out({ssize_t(v.engines.size()), ssize_t(NUM_ACTIONS)});
                               std::memcpy(out.mutable_data(), v.last_probs.data(), sizeof(float) * v.last_probs.size());
                               return out;
                             })
      .def("engine_state", [](const VecEnv& v, int i) {
        const Engine& e = v.engines.at(i);
        py::dict d;
        d["hands"] = std::vector<std::vector<int>>{{e.hands[0][0], e.hands[0][1]}, {e.hands[1][0], e.hands[1][1]}};
        d["board"] = std::vector<int>(e.board, e.board + 5);
        d["stacks"] = std::vector<int>{e.stacks[0], e.stacks[1]};
        d["bets"] = std::vector<int>{e.bets[0], e.bets[1]};
        d["stage_bets"] = std::vector<int>{e.stage_bets[0], e.stage_bets[1]};
        d["pot"] = e.pot;
        d["stage"] = e.stage;
        d["current"] = e.current;
        d["folded"] = e.folded;
        d["done"] = e.done;
        d["rewards"] = std::vector<int>{e.rewards[0], e.rewards[1]};
        return d;
      });
}
