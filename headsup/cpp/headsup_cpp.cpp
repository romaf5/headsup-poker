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

// observation layout (see headsup/engine.py): 31 original features + the DeepCFR bet history
constexpr int OBS_DIM_AGGREGATED = 31;
constexpr int HISTORY_ROUNDS = 4;
constexpr int HISTORY_SLOTS = 6;
constexpr int HISTORY_OFFSET = OBS_DIM_AGGREGATED;
constexpr int HISTORY_DIM = HISTORY_ROUNDS * HISTORY_SLOTS * 2;
constexpr int OBS_DIM_HISTORY = HISTORY_OFFSET + HISTORY_DIM;  // 79
constexpr int RAISES_INDEX = OBS_DIM_HISTORY;                  // consecutive raises on this street
constexpr int OBS_DIM = OBS_DIM_HISTORY + 1;                   // 80
constexpr int MAX_ACTIONS = 8;  // FOLD, CHECK_CALL, up to 5 raise sizes, ALL_IN (headsup/game.py: MAX_ACTIONS)
constexpr int NUM_CARDS = 52;
enum Action { FOLD = 0, CHECK_CALL = 1, RAISE = 2 };  // raise sizes are 2 .. num_actions-2, all-in = num_actions-1
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
  std::vector<double> bet_sizes = {-1.0};  // -1 = "min" (call + big blind), else fraction of the pot after calling
  bool mask_redundant = false;             // hide raises that duplicate another action (see headsup/game.py)
  int num_actions() const { return int(bet_sizes.size()) + 3; }
  int all_in() const { return num_actions() - 1; }
  bool is_raise(int a) const { return a >= 2 && a < all_in(); }
};

struct Engine {
  EngineConfig cfg;
  int hands[2][2] = {{0, 1}, {2, 3}};
  int board[5] = {4, 5, 6, 7, 8};
  int stacks[2] = {0, 0}, bets[2] = {0, 0}, stage_bets[2] = {0, 0};
  int pot = 0, stage = PREFLOP, current = 0, folded = -1, acted = 0, consecutive_raises = 0;
  int rewards[2] = {0, 0};
  bool done = true;
  // bet history: chips / pot-before-the-action of the first HISTORY_SLOTS actions per street
  float hist_size[HISTORY_ROUNDS][HISTORY_SLOTS] = {};
  int hist_n[HISTORY_ROUNDS] = {0, 0, 0, 0};
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
    std::memset(hist_size, 0, sizeof(hist_size));
    hist_n[0] = hist_n[1] = hist_n[2] = hist_n[3] = 0;
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
    // canonical card order (hole cards and flop sorted by id), as in the Python engine
    put(0, std::min(hands[p][0], hands[p][1]));
    put(1, std::max(hands[p][0], hands[p][1]));
    const int nb = BOARD_CARDS_BY_STAGE[stage];
    if (nb) {
      int flop[3] = {board[0], board[1], board[2]};
      std::sort(flop, flop + 3);
      for (int i = 0; i < 3; ++i) put(2 + i, flop[i]);
      for (int i = 3; i < nb; ++i) put(2 + i, board[i]);
    }
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
    obs[RAISES_INDEX] = float(consecutive_raises);
    for (int r = 0; r < HISTORY_ROUNDS; ++r) {
      const int n = std::min(hist_n[r], HISTORY_SLOTS);
      float* slot = obs + HISTORY_OFFSET + 2 * HISTORY_SLOTS * r;
      for (int k = 0; k < n; ++k) {
        slot[2 * k] = hist_size[r][k];
        slot[2 * k + 1] = 1.0f;
      }
    }
  }

  void record(int amount) {
    const int k = hist_n[stage];
    if (k < HISTORY_SLOTS) hist_size[stage][k] = float(amount) / float(pot);
    hist_n[stage] = k + 1;
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
  int to_call() const { return stage_bets[1 - current] - stage_bets[current]; }
  int num_actions() const { return cfg.num_actions(); }

  // chips the current player puts in for raise action a (2 .. all_in-1), capped by the stack
  int raise_amount(int a) const {
    const int call = to_call(), stack = stacks[current];
    const double size = cfg.bet_sizes[a - 2];
    const int min_raise = call + cfg.big_blind;
    int amount = min_raise;
    if (size >= 0) amount = std::max(min_raise, call + int(std::lround(size * double(pot + call))));
    return std::min(amount, stack);
  }

  // mask[a] = the action is meaningful here; twin[a] = the action a redundant one duplicates (else a)
  void legal_mask(bool* mask, int* twin = nullptr) const {
    const int n = num_actions(), ai = cfg.all_in();
    for (int a = 0; a < n; ++a) {
      mask[a] = a != FOLD || fold_allowed();
      if (twin) twin[a] = a;
    }
    if (twin && !fold_allowed()) twin[FOLD] = CHECK_CALL;
    if (!cfg.mask_redundant) return;
    const int stack = stacks[current];
    int amounts[MAX_ACTIONS];
    for (int a = 2; a < ai; ++a) {
      const int amount = raise_amount(a);
      amounts[a] = amount;
      int dup = -1;
      for (int b = 2; b < a; ++b)
        if (amounts[b] == amount) { dup = b; break; }
      if (amount >= stack || consecutive_raises + 1 >= cfg.raise_cap) {
        mask[a] = false;
        if (twin) twin[a] = ai;
      } else if (dup >= 0) {
        mask[a] = false;
        if (twin) twin[a] = twin[dup];
      }
    }
  }

  bool step(int action) {
    const int p = current, o = 1 - p;
    const int ai = cfg.all_in();
    if (action == FOLD && stage_bets[o] == stage_bets[p]) action = CHECK_CALL;  // nothing to call: fold is a dominated check
    int amount = 0;
    if (cfg.is_raise(action)) {
      amount = raise_amount(action);
      if (++consecutive_raises >= cfg.raise_cap) action = ai;
    } else {
      consecutive_raises = 0;
    }
    if (action == FOLD) {
      record(0);
      folded = p;
      rewards[o] = bets[p];
      rewards[p] = -bets[p];
      done = true;
      return true;
    }
    if (action == CHECK_CALL)
      amount = std::min(stage_bets[o] - stage_bets[p], stacks[p]);
    else if (action == ai)
      amount = stacks[p];
    record(amount);
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
  std::vector<float> wt, b;  // wt is the transposed weight, (in, out) row-major
  // y = b + W x as a sequence of axpys over the inputs: the inner loop is a contiguous,
  // reduction-free vector update (auto-vectorised under strict FP), and zero inputs (the
  // usual case after a ReLU) are skipped
  void apply(const float* x, float* y) const {
    std::memcpy(y, b.data(), sizeof(float) * out);
    for (int i = 0; i < in; ++i) {
      const float xi = x[i];
      if (xi == 0.0f) continue;
      const float* row = wt.data() + size_t(i) * out;
      for (int o = 0; o < out; ++o) y[o] += row[o] * xi;
    }
  }
};

inline void relu(float* x, int n) {
  for (int i = 0; i < n; ++i) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

enum Features { AGGREGATED = 0, HISTORY = 1, BOTH_FEATURES = 2 };
enum Arch { CURRENT = 0, PAPER = 1 };
enum Cards { EMBED = 0, ONEHOT = 1 };
constexpr int MAX_DIM = 256;                       // widest supported hidden layer
constexpr int CARD_CLASSES = NUM_CARDS + 1;        // one-hot classes per card slot (0 = no card)
constexpr int MAX_BET_FEATURES = 8 + HISTORY_DIM + 2 + 1;
constexpr int GROUP_OF_SLOT[7] = {0, 0, 1, 1, 1, 2, 3};  // hole, flop, turn, river

// Mirrors headsup.model.BaseModel for every variant (see the docstring there).
struct Model {
  int dim = 64, obs_dim = OBS_DIM_AGGREGATED, num_actions = 4;
  int features = AGGREGATED, arch = CURRENT, cards = EMBED;
  bool rm_argmax = false;  // regret-matching fallback: highest advantage instead of uniform
  std::vector<int> bet_index;
  // embed: rank/suit/card tables, shared (index 0) or one set per card group (paper arch)
  std::vector<float> rank_emb[4], suit_emb[4], card_emb[4];
  bool per_group = false;
  std::vector<float> onehot_t;  // (7 * CARD_CLASSES, dim): transposed weight of the one-hot layer
  std::vector<float> onehot_b;
  Linear card_fc[3];  // card_fc[0] unused for one-hot cards
  std::vector<float> stage_emb, first_emb;
  Linear stage_fc[2], bets_fc[2], comb[3], head;

  void forward(const float* obs, float* out) const {
    const int D = dim;
    float c1[MAX_DIM], c2[MAX_DIM];
    // 1. card branch
    if (cards == EMBED) {
      float x[4 * MAX_DIM];
      std::memset(x, 0, sizeof(float) * 4 * D);
      for (int s = 0; s < 7; ++s) {
        const int g = GROUP_OF_SLOT[s], tbl = per_group ? g : 0;
        const int r = int(obs[3 * s]), su = int(obs[3 * s + 1]), c = int(obs[3 * s + 2]);
        const float* er = rank_emb[tbl].data() + size_t(r) * D;
        const float* es = suit_emb[tbl].data() + size_t(su) * D;
        const float* ec = card_emb[tbl].data() + size_t(c) * D;
        float* dst = x + g * D;
        for (int i = 0; i < D; ++i) dst[i] += er[i] + es[i] + ec[i];
      }
      card_fc[0].apply(x, c1);
    } else {  // Linear on concatenated one-hot cards == bias + sum of the selected weight columns
      std::memcpy(c1, onehot_b.data(), sizeof(float) * D);
      for (int s = 0; s < 7; ++s) {
        const float* row = onehot_t.data() + (size_t(s) * CARD_CLASSES + int(obs[3 * s + 2])) * D;
        for (int i = 0; i < D; ++i) c1[i] += row[i];
      }
    }
    relu(c1, D);
    card_fc[1].apply(c1, c2);
    relu(c2, D);
    card_fc[2].apply(c2, c1);
    relu(c1, D);  // c1 = card features

    // 2. stage / position branch (current arch only)
    float s2[MAX_DIM];
    if (arch == CURRENT) {
      float se[2 * MAX_DIM], s1[MAX_DIM];
      const int st = int(obs[21]), fa = int(obs[22]);
      std::memcpy(se, stage_emb.data() + size_t(st) * D, sizeof(float) * D);
      std::memcpy(se + D, first_emb.data() + size_t(fa) * D, sizeof(float) * D);
      stage_fc[0].apply(se, s1);
      relu(s1, D);
      stage_fc[1].apply(s1, s2);
      relu(s2, D);  // s2 = stage features
    }

    // 3. bet branch
    float bin[MAX_BET_FEATURES], b1[MAX_DIM], b2[MAX_DIM];
    for (size_t j = 0; j < bet_index.size(); ++j) bin[j] = obs[bet_index[j]];
    bets_fc[0].apply(bin, b1);
    relu(b1, D);
    bets_fc[1].apply(b1, b2);
    for (int i = 0; i < D; ++i) b2[i] += b1[i];
    relu(b2, D);  // b2 = bet features

    // 4. trunk
    float z[3 * MAX_DIM];
    std::memcpy(z, c1, sizeof(float) * D);
    int off = D;
    if (arch == CURRENT) {
      std::memcpy(z + off, s2, sizeof(float) * D);
      off += D;
    }
    std::memcpy(z + off, b2, sizeof(float) * D);
    float z1[MAX_DIM], z2[MAX_DIM];
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
  const std::vector<float> w = to_vec(d, (prefix + ".weight").c_str(), size_t(in) * out);  // (out, in)
  l.wt.assign(size_t(in) * out, 0.0f);
  for (int o = 0; o < out; ++o)
    for (int i = 0; i < in; ++i) l.wt[size_t(i) * out + o] = w[size_t(o) * in + i];
  l.b = to_vec(d, (prefix + ".bias").c_str(), out);
  return l;
}

std::vector<int> bet_feature_indices(int features, int arch) {
  std::vector<int> idx;
  if (features == AGGREGATED || features == BOTH_FEATURES)
    for (int i = 23; i < OBS_DIM_AGGREGATED; ++i) idx.push_back(i);
  if (features == HISTORY || features == BOTH_FEATURES) {
    for (int i = HISTORY_OFFSET; i < HISTORY_OFFSET + HISTORY_DIM; ++i) idx.push_back(i);
    if (features == HISTORY) {
      idx.push_back(28);
      idx.push_back(29);
    }
  }
  if (arch == PAPER) idx.push_back(22);
  return idx;
}

// `d` = BaseModel.numpy_weights(): float32 arrays plus a "config" dict.
std::shared_ptr<Model> model_from_dict(const py::dict& d) {
  auto m = std::make_shared<Model>();
  if (!d.contains("config")) throw std::runtime_error("model weights need a 'config' entry (see headsup.native.make_model)");
  py::dict cfg = d["config"].cast<py::dict>();
  const std::string features = cfg["features"].cast<std::string>();
  const std::string arch = cfg["arch"].cast<std::string>();
  const std::string cards = cfg["cards"].cast<std::string>();
  const std::string rm = cfg.contains("rm_fallback") ? cfg["rm_fallback"].cast<std::string>() : "uniform";
  m->dim = cfg["dim"].cast<int>();
  if (m->dim < 2 || m->dim > MAX_DIM) throw std::runtime_error("dim must be in [2, " + std::to_string(MAX_DIM) + "]");
  m->features = features == "aggregated" ? AGGREGATED : features == "history" ? HISTORY : features == "both" ? BOTH_FEATURES : -1;
  m->arch = arch == "current" ? CURRENT : arch == "paper" ? PAPER : -1;
  m->cards = cards == "embed" ? EMBED : cards == "onehot" ? ONEHOT : -1;
  if (m->features < 0 || m->arch < 0 || m->cards < 0) throw std::runtime_error("unknown model config values");
  m->rm_argmax = rm == "argmax";
  m->num_actions = 4;
  if (cfg.contains("game")) {
    py::dict game = cfg["game"].cast<py::dict>();
    m->num_actions = int(py::len(game["bet_sizes"])) + 3;
  }
  if (m->num_actions < 3 || m->num_actions > MAX_ACTIONS) throw std::runtime_error("unsupported number of actions");
  m->obs_dim = m->features == AGGREGATED ? OBS_DIM_AGGREGATED : OBS_DIM_HISTORY;
  m->bet_index = bet_feature_indices(m->features, m->arch);
  m->per_group = m->arch == PAPER;
  const int D = m->dim;
  if (m->cards == EMBED) {
    if (m->per_group) {
      for (int g = 0; g < 4; ++g) {
        const std::string pre = "card_model.group_embeddings." + std::to_string(g) + ".";
        m->rank_emb[g] = to_vec(d, (pre + "rank_embedding.weight").c_str(), 14 * D);
        m->suit_emb[g] = to_vec(d, (pre + "suit_embedding.weight").c_str(), 5 * D);
        m->card_emb[g] = to_vec(d, (pre + "card_embedding.weight").c_str(), 53 * D);
      }
    } else {
      m->rank_emb[0] = to_vec(d, "card_model.cards_embeddings.rank_embedding.weight", 14 * D);
      m->suit_emb[0] = to_vec(d, "card_model.cards_embeddings.suit_embedding.weight", 5 * D);
      m->card_emb[0] = to_vec(d, "card_model.cards_embeddings.card_embedding.weight", 53 * D);
    }
    m->card_fc[0] = to_linear(d, "card_model.fc1", 4 * D, D);
  } else {
    Linear oh = to_linear(d, "card_model.onehot", 7 * CARD_CLASSES, D);
    m->onehot_t = std::move(oh.wt);  // (7 * CARD_CLASSES, D): row i = weights of one-hot input i
    m->onehot_b = oh.b;
  }
  m->card_fc[1] = to_linear(d, "card_model.fc2", D, D);
  m->card_fc[2] = to_linear(d, "card_model.fc3", D, D);
  if (m->arch == CURRENT) {
    m->stage_emb = to_vec(d, "stage_and_order_model.stage_embedding.weight", 4 * D);
    m->first_emb = to_vec(d, "stage_and_order_model.first_to_act_embedding.weight", 2 * D);
    m->stage_fc[0] = to_linear(d, "stage_and_order_model.fc1", 2 * D, D);
    m->stage_fc[1] = to_linear(d, "stage_and_order_model.fc2", D, D);
  }
  m->bets_fc[0] = to_linear(d, "bets_model.fc1", int(m->bet_index.size()), D);
  m->bets_fc[1] = to_linear(d, "bets_model.fc2", D, D);
  m->comb[0] = to_linear(d, "comb1", (m->arch == CURRENT ? 3 : 2) * D, D);
  m->comb[1] = to_linear(d, "comb2", D, D);
  m->comb[2] = to_linear(d, "comb3", D, D);
  m->head = to_linear(d, "action_head", D, m->num_actions);
  return m;
}

// regret matching over the legal actions; when no legal advantage is positive: uniform over the
// legal actions, or (argmax_fallback, DeepCFR paper) the highest legal advantage with probability 1
inline void regret_matching(const float* adv, float* sigma, const bool* legal, int n, bool argmax_fallback = false) {
  float total = 0.0f;
  for (int a = 0; a < n; ++a) {
    sigma[a] = (legal[a] && adv[a] > 0.0f) ? adv[a] : 0.0f;
    total += sigma[a];
  }
  if (total <= 1e-6f) {
    if (argmax_fallback) {
      int best = -1;
      for (int a = 0; a < n; ++a)
        if (legal[a] && (best < 0 || adv[a] > adv[best])) best = a;
      for (int a = 0; a < n; ++a) sigma[a] = a == best ? 1.0f : 0.0f;
    } else {
      int cnt = 0;
      for (int a = 0; a < n; ++a) cnt += legal[a];
      for (int a = 0; a < n; ++a) sigma[a] = legal[a] ? 1.0f / cnt : 0.0f;
    }
  } else {
    for (int a = 0; a < n; ++a) sigma[a] /= total;
  }
}

inline void softmax(const float* logits, float* p, int n) {
  float m = logits[0];
  for (int a = 1; a < n; ++a) m = std::max(m, logits[a]);
  float s = 0.0f;
  for (int a = 0; a < n; ++a) {
    p[a] = std::exp(logits[a] - m);
    s += p[a];
  }
  for (int a = 0; a < n; ++a) p[a] /= s;
}

template <class RNG>
inline int sample(const float* p, int n, RNG& rng) {
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  const float u = u01(rng);
  float acc = 0.0f;
  for (int a = 0; a < n - 1; ++a) {
    acc += p[a];
    if (u < acc) return a;
  }
  return n - 1;
}

// ------------------------------------------------------------------ MCCFR traversal
struct Memory {
  int obs_dim = OBS_DIM;  // width stored per sample (= the networks' input width)
  int num_actions = 4;
  std::vector<float> obs, t, target;
  void add(const float* o, float tt, const float* tg) {
    obs.insert(obs.end(), o, o + obs_dim);
    t.push_back(tt);
    target.insert(target.end(), tg, tg + num_actions);
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
    const int p = e.current, n = e.num_actions();
    float obs[OBS_DIM], values[MAX_ACTIONS], sigma[MAX_ACTIONS];
    bool legal[MAX_ACTIONS];
    int twin[MAX_ACTIONS];
    e.observation(-1, obs);
    nets[p]->forward(obs, values);
    e.legal_mask(legal, twin);
    regret_matching(values, sigma, legal, n, nets[p]->rm_argmax);
    ++nodes;
    if (p == traverser) {
      float va[MAX_ACTIONS];
      for (int a = 0; a < n; ++a) {
        if (!legal[a]) continue;  // duplicates another action: filled in below
        if (a + 1 < n) {
          Engine child = e;
          child.step(a);
          va[a] = traverse(child);
        } else {
          e.step(a);
          va[a] = traverse(e);
        }
      }
      for (int a = 0; a < n; ++a)
        if (!legal[a]) va[a] = va[twin[a]];
      float mean = 0.0f;
      for (int a = 0; a < n; ++a) mean += sigma[a] * va[a];
      for (int a = 0; a < n; ++a) va[a] -= mean;
      adv.add(obs, t, va);
      return mean;
    }
    strat.add(obs, t, sigma);
    e.step(sample(sigma, n, rng));
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
  if (net0->obs_dim != net1->obs_dim) throw std::runtime_error("both networks must read the same observation width");
  if (net0->num_actions != cfg.num_actions() || net1->num_actions != cfg.num_actions())
    throw std::runtime_error("the networks' action heads do not match the game's number of actions");
  tr.adv.obs_dim = tr.strat.obs_dim = net0->obs_dim;
  tr.adv.num_actions = tr.strat.num_actions = cfg.num_actions();
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
  const ssize_t od = tr.adv.obs_dim, nact = cfg.num_actions();
  return py::make_tuple(to_array(tr.adv.obs, na, od), to_array(tr.adv.t, na, 1),
                        to_array(tr.adv.target, na, nact), to_array(tr.strat.obs, ns, od),
                        to_array(tr.strat.t, ns, 1), to_array(tr.strat.target, ns, nact), tr.nodes);
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
  int num_actions;
  std::vector<float> last_probs;  // (num_envs, num_actions) of the opponent's last decision per table
  long hands_completed = 0;

  VecEnv(int n, uint64_t seed, EngineConfig cfg, bool alternate_seats)
      : engines(size_t(n), Engine(cfg)), agent_seat(size_t(n)), alternate(alternate_seats), rng(seed),
        num_actions(cfg.num_actions()), last_probs(size_t(n) * cfg.num_actions(), 1.0f / cfg.num_actions()) {
    for (int i = 0; i < n; ++i) agent_seat[i] = (i % 2) ^ 1;  // flipped on first reset
  }

  int opponent_action(int i, const float* obs) {
    const Engine& e = engines[i];
    const int n = num_actions;
    switch (opp_kind) {
      case RANDOM: {  // uniform over the legal actions
        bool legal[MAX_ACTIONS];
        e.legal_mask(legal);
        int cnt = 0;
        for (int a = 0; a < n; ++a) cnt += legal[a];
        std::uniform_int_distribution<int> d(0, cnt - 1);
        int k = d(rng);
        for (int a = 0; a < n; ++a)
          if (legal[a] && k-- == 0) return a;
        return CHECK_CALL;
      }
      case CALL:
        return CHECK_CALL;
      case ALLIN:
        return e.cfg.all_in();
      case RAISE_:
        return RAISE;
      case MODEL: {
        float logits[MAX_ACTIONS];
        bool legal[MAX_ACTIONS];
        float* p = last_probs.data() + size_t(i) * n;
        opp_model->forward(obs, logits);
        e.legal_mask(legal);
        for (int a = 0; a < n; ++a)
          if (!legal[a]) logits[a] = -1e30f;
        softmax(logits, p, n);
        if (opp_deterministic) return int(std::max_element(p, p + n) - p);
        return sample(p, n, rng);
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


// ------------------------------------------------------------------ equity vs every opponent hand (LBR)
constexpr int NUM_COMBOS = 1326;

inline int combo_index(int a, int b) {  // a < b
  return a * NUM_CARDS - a * (a + 1) / 2 + (b - a - 1);
}

// equity[h] = P(win) + P(tie)/2 of hole cards (c0, c1) against opponent combo h on `board`
// (0..5 known cards, the rest dealt uniformly: every runout when there are at most `max_exact`
// of them, otherwise `samples` Monte-Carlo runouts).  Combos that overlap the hero's cards or
// the board get -1.  Runouts that collide with a combo are skipped for that combo.
void equity_vs_all(int c0, int c1, const int* board, int n_board, int samples, int max_exact, uint64_t seed, float* out) {
  bool used[NUM_CARDS] = {};
  used[c0] = used[c1] = true;
  for (int i = 0; i < n_board; ++i) used[board[i]] = true;
  int deck[NUM_CARDS], n_deck = 0;
  for (int c = 0; c < NUM_CARDS; ++c)
    if (!used[c]) deck[n_deck++] = c;
  const int missing = 5 - n_board;
  std::vector<double> win(NUM_COMBOS, 0.0), cnt(NUM_COMBOS, 0.0);
  std::vector<int> combo_a, combo_b;  // valid combos
  combo_a.reserve(NUM_COMBOS);
  combo_b.reserve(NUM_COMBOS);
  for (int a = 0; a < NUM_CARDS; ++a)
    for (int b = a + 1; b < NUM_CARDS; ++b)
      if (!used[a] && !used[b]) {
        combo_a.push_back(a);
        combo_b.push_back(b);
      }
  auto evaluate_runout = [&](const int* extra) {
    int full[5];
    for (int i = 0; i < n_board; ++i) full[i] = board[i];
    for (int i = 0; i < missing; ++i) full[n_board + i] = extra[i];
    bool on_board[NUM_CARDS] = {};
    for (int i = 0; i < missing; ++i) on_board[extra[i]] = true;
    int mine[7] = {c0, c1, full[0], full[1], full[2], full[3], full[4]};
    const int s_me = eval7(mine);
    for (size_t k = 0; k < combo_a.size(); ++k) {
      const int a = combo_a[k], b = combo_b[k];
      if (on_board[a] || on_board[b]) continue;
      int his[7] = {a, b, full[0], full[1], full[2], full[3], full[4]};
      const int s = eval7(his);
      const int idx = combo_index(a, b);
      cnt[idx] += 1.0;
      if (s_me < s) win[idx] += 1.0;
      else if (s_me == s) win[idx] += 0.5;
    }
  };
  if (missing == 0) {
    evaluate_runout(nullptr);
  } else {
    // number of runouts: C(n_deck, missing)
    double n_runouts = 1.0;
    for (int i = 0; i < missing; ++i) n_runouts = n_runouts * (n_deck - i) / (i + 1);
    if (n_runouts <= max_exact) {
      int idx[5];
      for (int i = 0; i < missing; ++i) idx[i] = i;
      while (true) {
        int extra[5];
        for (int i = 0; i < missing; ++i) extra[i] = deck[idx[i]];
        evaluate_runout(extra);
        int i = missing - 1;
        while (i >= 0 && idx[i] == n_deck - missing + i) --i;
        if (i < 0) break;
        ++idx[i];
        for (int j = i + 1; j < missing; ++j) idx[j] = idx[j - 1] + 1;
      }
    } else {
      std::mt19937_64 rng(seed);
      int pool[NUM_CARDS];
      for (int s_i = 0; s_i < samples; ++s_i) {
        std::memcpy(pool, deck, sizeof(int) * n_deck);
        int extra[5];
        for (int i = 0; i < missing; ++i) {  // partial Fisher-Yates
          std::uniform_int_distribution<int> d(i, n_deck - 1);
          const int j = d(rng);
          std::swap(pool[i], pool[j]);
          extra[i] = pool[i];
        }
        evaluate_runout(extra);
      }
    }
  }
  for (int h = 0; h < NUM_COMBOS; ++h) out[h] = cnt[h] > 0 ? float(win[h] / cnt[h]) : -1.0f;
}

// ------------------------------------------------------------------ micro benchmarks (for profiling)
double bench_engine(int hands, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Engine e;
  std::uniform_int_distribution<int> d(0, e.num_actions() - 1);
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
  float obs[OBS_DIM] = {0}, out[MAX_ACTIONS];
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
  m.attr("OBS_DIM") = OBS_DIM;
  m.attr("OBS_DIM_AGGREGATED") = OBS_DIM_AGGREGATED;
  m.attr("OBS_DIM_HISTORY") = OBS_DIM_HISTORY;
  m.attr("MAX_ACTIONS") = MAX_ACTIONS;
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
      .def_readwrite("raise_cap", &EngineConfig::raise_cap)
      .def_readwrite("bet_sizes", &EngineConfig::bet_sizes)
      .def_readwrite("mask_redundant", &EngineConfig::mask_redundant)
      .def_property_readonly("num_actions", &EngineConfig::num_actions);

  py::class_<Engine>(m, "Engine")
      .def(py::init<EngineConfig>(), py::arg("cfg") = EngineConfig())
      .def("reset",
           [](Engine& e, std::vector<int> deck) {
             if (deck.size() < 9) throw std::runtime_error("deck needs >= 9 cards");
             e.reset(deck.data());
           })
      .def("step", [](Engine& e, int a) {
        if (a < 0 || a >= e.num_actions()) throw std::runtime_error("invalid action");
        return e.step(a);
      })
      .def("legal_mask", [](const Engine& e) {
        bool legal[MAX_ACTIONS];
        e.legal_mask(legal);
        return std::vector<bool>(legal, legal + e.num_actions());
      })
      .def("raise_amount", [](const Engine& e, int a) { return e.raise_amount(a); })
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
      .def_property_readonly("obs_dim", [](const Model& mm) { return mm.obs_dim; })
      .def_property_readonly("num_actions", [](const Model& mm) { return mm.num_actions; })
      .def_property_readonly("dim", [](const Model& mm) { return mm.dim; })
      .def_property_readonly("rm_argmax", [](const Model& mm) { return mm.rm_argmax; })
      .def("forward", [](const Model& mm, py::array_t<float, py::array::c_style | py::array::forcecast> obs) {
        if (obs.ndim() == 1) {
          py::array_t<float> out(std::vector<ssize_t>{mm.num_actions});
          mm.forward(obs.data(), out.mutable_data());
          return out;
        }
        const ssize_t n = obs.shape(0);
        py::array_t<float> out({n, ssize_t(mm.num_actions)});
        auto o = out.mutable_unchecked<2>();
        auto in = obs.unchecked<2>();
        for (ssize_t i = 0; i < n; ++i) mm.forward(in.data(i, 0), o.mutable_data(i, 0));
        return out;
      });

  m.def(
      "equity_vs_all",
      [](int c0, int c1, std::vector<int> board, int samples, int max_exact, uint64_t seed) {
        if (board.size() > 5) throw std::runtime_error("board has at most 5 cards");
        py::array_t<float> out(std::vector<ssize_t>{NUM_COMBOS});
        float* ptr = out.mutable_data();
        {
          py::gil_scoped_release release;
          equity_vs_all(c0, c1, board.data(), int(board.size()), samples, max_exact, seed, ptr);
        }
        return out;
      },
      py::arg("c0"), py::arg("c1"), py::arg("board"), py::arg("samples") = 200, py::arg("max_exact") = 1200,
      py::arg("seed") = 0,
      "P(win) + P(tie)/2 of (c0, c1) vs every opponent combo (index = combo_index(a, b), a < b); -1 = blocked");
  m.def("combo_index", &combo_index);
  m.attr("NUM_COMBOS") = NUM_COMBOS;

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
                               py::array_t<float> out({ssize_t(v.engines.size()), ssize_t(v.num_actions)});
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
