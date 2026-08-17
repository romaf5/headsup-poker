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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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

// best 5-card hand of n = 5..7 cards (treys-compatible), for games that show down on fewer board cards
int eval_best(const int* cards, int n) {
  if (n == 7) return eval7(cards);
  uint32_t c[7];
  for (int i = 0; i < n; ++i) c[i] = treys_card(cards[i]);
  if (n == 5) return eval5(c[0], c[1], c[2], c[3], c[4]);
  int best = 7462;  // n == 6: leave one out
  for (int skip = 0; skip < 6; ++skip) {
    uint32_t h[5];
    int k = 0;
    for (int i = 0; i < 6; ++i)
      if (i != skip) h[k++] = c[i];
    best = std::min(best, eval5(h[0], h[1], h[2], h[3], h[4]));
  }
  return best;
}

// ------------------------------------------------------------------ engine
struct EngineConfig {
  int stack_size = 100;
  int small_blind = 1;
  int big_blind = 2;
  int raise_cap = 3;
  std::vector<double> bet_sizes = {-1.0};  // -1 = "min" (call + big blind), -2 = limit increment, else pot fraction
  bool mask_redundant = false;             // hide raises that duplicate another action (see headsup/game.py)
  std::vector<int> limit;                  // limit poker: fixed raise increment per round (empty = no-limit sizes)
  std::vector<int> raise_caps;             // per-round raise caps (empty: raise_cap for every round)
  int num_rounds = 4;                      // betting rounds; the showdown uses the board of the last one
  bool has_all_in = true;                  // whether the ALL_IN action exists (limit games: no)
  int num_raises() const { return int(bet_sizes.size()); }
  int num_actions() const { return 2 + num_raises() + (has_all_in ? 1 : 0); }
  int all_in() const { return has_all_in ? 2 + num_raises() : -1; }
  bool is_raise(int a) const { return a >= 2 && a < 2 + num_raises(); }
  int cap(int round) const { return raise_caps.empty() ? raise_cap : raise_caps[round]; }
  int showdown_cards() const { return BOARD_CARDS_BY_STAGE[num_rounds - 1]; }
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
    const int nb = cfg.showdown_cards();
    int c0[7] = {hands[0][0], hands[0][1], board[0], board[1], board[2], board[3], board[4]};
    int c1[7] = {hands[1][0], hands[1][1], board[0], board[1], board[2], board[3], board[4]};
    const int s0 = eval_best(c0, 2 + nb), s1 = eval_best(c1, 2 + nb);
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

  // chips the current player puts in for raise action a (2 .. 2+K-1), capped by the stack
  int raise_amount(int a) const {
    const int call = to_call(), stack = stacks[current];
    const double size = cfg.bet_sizes[a - 2];
    if (size == -2.0) return std::min(call + cfg.limit[stage], stack);  // limit: fixed increment per round
    const int min_raise = call + cfg.big_blind;
    int amount = min_raise;
    if (size >= 0) amount = std::max(min_raise, call + int(std::lround(size * double(pot + call))));
    return std::min(amount, stack);
  }

  // mask[a] = the action is meaningful here; twin[a] = the action a redundant one duplicates (else a)
  void legal_mask(bool* mask, int* twin = nullptr) const {
    const int n = num_actions(), ai = cfg.all_in(), nr = 2 + cfg.num_raises();
    for (int a = 0; a < n; ++a) {
      mask[a] = a != FOLD || fold_allowed();
      if (twin) twin[a] = a;
    }
    if (twin && !fold_allowed()) twin[FOLD] = CHECK_CALL;
    if (!cfg.mask_redundant && cfg.has_all_in) return;
    const int stack = stacks[current], call = to_call();
    // no-limit: the cap-th raise in a row is executed as an all-in; limit: no raise past the cap
    const bool capped = cfg.has_all_in ? consecutive_raises + 1 >= cfg.cap(stage) : consecutive_raises >= cfg.cap(stage);
    const int collapse = cfg.has_all_in ? ai : CHECK_CALL;
    int amounts[MAX_ACTIONS];
    for (int a = 2; a < nr; ++a) {
      const int amount = raise_amount(a);
      amounts[a] = amount;
      int dup = -1;
      for (int b = 2; b < a; ++b)
        if (amounts[b] == amount) { dup = b; break; }
      if (capped || (amount >= stack && cfg.has_all_in) || stack <= call) {
        mask[a] = false;
        if (twin) twin[a] = collapse;
      } else if (cfg.mask_redundant && dup >= 0) {
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
      ++consecutive_raises;
      const int cap = cfg.cap(stage);
      if (cfg.has_all_in && consecutive_raises >= cap) {
        action = ai;  // no-limit: the cap-th raise in a row becomes an all-in
      } else if (!cfg.has_all_in && consecutive_raises > cap) {
        --consecutive_raises;  // limit: no raise past the cap - executed as a call
        action = CHECK_CALL;
      }
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
    else if (cfg.has_all_in && action == ai)
      amount = stacks[p];
    record(amount);
    bets[p] += amount;
    stage_bets[p] += amount;
    stacks[p] -= amount;
    pot += amount;
    acted |= 1 << p;
    current = o;
    if (street_finished()) {
      if (stage == cfg.num_rounds - 1 || std::min(stacks[0], stacks[1]) == 0) {
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
constexpr int GROUP_OF_SLOT[9] = {0, 0, 1, 1, 1, 2, 3, 4, 4};  // hole, flop, turn, river, (opponent's hole cards)
constexpr int OPP_CARDS_OFFSET = OBS_DIM;                      // history inputs: opponent's cards after the observation
constexpr int OBS_DIM_WITH_OPP = OBS_DIM + 6;
inline int slot_offset(int s) { return s < 7 ? 3 * s : OPP_CARDS_OFFSET + 3 * (s - 7); }

// Mirrors headsup.model.BaseModel for every variant (see the docstring there).
struct Model {
  int dim = 64, obs_dim = OBS_DIM_AGGREGATED, num_actions = 4;
  bool opp_cards = false;  // history-input network: + the opponent's hole cards as a fifth card group
  int n_slots = 7, n_groups = 4;
  int features = AGGREGATED, arch = CURRENT, cards = EMBED;
  bool rm_argmax = false;  // regret-matching fallback: highest advantage instead of uniform
  std::vector<int> bet_index;
  // embed: rank/suit/card tables, shared (index 0) or one set per card group (paper arch)
  std::vector<float> rank_emb[5], suit_emb[5], card_emb[5];
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
      float x[5 * MAX_DIM];
      std::memset(x, 0, sizeof(float) * n_groups * D);
      for (int s = 0; s < n_slots; ++s) {
        const int g = GROUP_OF_SLOT[s], tbl = per_group ? g : 0, o = slot_offset(s);
        const int r = int(obs[o]), su = int(obs[o + 1]), c = int(obs[o + 2]);
        const float* er = rank_emb[tbl].data() + size_t(r) * D;
        const float* es = suit_emb[tbl].data() + size_t(su) * D;
        const float* ec = card_emb[tbl].data() + size_t(c) * D;
        float* dst = x + g * D;
        for (int i = 0; i < D; ++i) dst[i] += er[i] + es[i] + ec[i];
      }
      card_fc[0].apply(x, c1);
    } else {  // Linear on concatenated one-hot cards == bias + sum of the selected weight columns
      std::memcpy(c1, onehot_b.data(), sizeof(float) * D);
      for (int s = 0; s < n_slots; ++s) {
        const float* row = onehot_t.data() + (size_t(s) * CARD_CLASSES + int(obs[slot_offset(s) + 2])) * D;
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
    const bool all_in = !game.contains("all_in") || game["all_in"].is_none() || game["all_in"].cast<bool>();
    m->num_actions = int(py::len(game["bet_sizes"])) + 2 + (all_in ? 1 : 0);
  }
  if (m->num_actions < 3 || m->num_actions > MAX_ACTIONS) throw std::runtime_error("unsupported number of actions");
  m->opp_cards = cfg.contains("opp_cards") && !cfg["opp_cards"].is_none() && cfg["opp_cards"].cast<bool>();
  m->n_slots = m->opp_cards ? 9 : 7;
  m->n_groups = m->opp_cards ? 5 : 4;
  m->obs_dim = m->opp_cards ? OBS_DIM_WITH_OPP : m->features == AGGREGATED ? OBS_DIM_AGGREGATED : OBS_DIM_HISTORY;
  m->bet_index = bet_feature_indices(m->features, m->arch);
  m->per_group = m->arch == PAPER;
  const int D = m->dim;
  if (m->cards == EMBED) {
    if (m->per_group) {
      for (int g = 0; g < m->n_groups; ++g) {
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
    m->card_fc[0] = to_linear(d, "card_model.fc1", m->n_groups * D, D);
  } else {
    Linear oh = to_linear(d, "card_model.onehot", m->n_slots * CARD_CLASSES, D);
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


// ------------------------------------------------------------------ DREAM / ESCHER trajectory samplers
// Outcome-sampling counterparts of the external-sampling traversal above, both driven by history
// value networks (Model with opp_cards: the observation of seat 0 plus seat 1's hole cards).
//   DREAM (Steinberger, Lerer & Brown 2020): one trajectory per traversal; the traverser explores
//   with xi = eps * uniform + (1 - eps) * sigma, the opponent plays sigma; the learned baseline
//   Q_p(h, a) replaces the unsampled actions' values (eq. 6-7): v~(a) = b(a) except the sampled one
//   b(a) + (v_child - b(a)) / xi(a); advantage samples (I_p, t / own sample reach, v~ - sigma.v~) and
//   expected-SARSA baseline targets (h, a, r + sigma(h').Q_p(h', .)).
//   ESCHER (McAleer et al. 2023): value trajectories under sigma give (h, a, u_0) regression rows for
//   the history value net; regret trajectories have the update player sample uniformly and the
//   opponent play sigma, with regrets q(h, .) - sigma.q(h, .) read off the value net (no importance
//   weights) and (I, t, sigma) rows for the average policy net.
inline void history_observation(const Engine& e, float* out) {
  e.observation(0, out);
  const int a = std::min(e.hands[1][0], e.hands[1][1]), b = std::max(e.hands[1][0], e.hands[1][1]);
  out[OPP_CARDS_OFFSET + 0] = float(a % 13 + 1); out[OPP_CARDS_OFFSET + 1] = float(a / 13 + 1); out[OPP_CARDS_OFFSET + 2] = float(a + 1);
  out[OPP_CARDS_OFFSET + 3] = float(b % 13 + 1); out[OPP_CARDS_OFFSET + 4] = float(b / 13 + 1); out[OPP_CARDS_OFFSET + 5] = float(b + 1);
}

struct TrajectorySampler {
  const Model* nets[2];
  const Model* value = nullptr;  // DREAM: baseline Q_traverser(h, a); ESCHER: history value q(h, a) of seat 0
  int traverser = 0;
  float t = 1.0f, epsilon = 0.5f;
  std::mt19937_64 rng;
  Memory adv, strat, val;  // val: history rows, t = action index, target[a] = regression target
  long nodes = 0;

  void sigma_at(const Engine& e, float* obs, float* sigma, bool* legal, int* twin) const {
    float values[MAX_ACTIONS];
    e.observation(-1, obs);
    nets[e.current]->forward(obs, values);
    e.legal_mask(legal, twin);
    regret_matching(values, sigma, legal, e.num_actions(), nets[e.current]->rm_argmax);
  }

  // -- DREAM ------------------------------------------------------------------------------
  float dream(Engine& e, double own_reach) {
    if (e.done) return float(e.rewards[traverser]);
    const int p = e.current, n = e.num_actions();
    float obs[OBS_DIM], sigma[MAX_ACTIONS], xi[MAX_ACTIONS], b[MAX_ACTIONS], hist[OBS_DIM_WITH_OPP];
    bool legal[MAX_ACTIONS];
    int twin[MAX_ACTIONS];
    sigma_at(e, obs, sigma, legal, twin);
    ++nodes;
    int n_legal = 0;
    for (int a = 0; a < n; ++a) n_legal += legal[a];
    for (int a = 0; a < n; ++a)
      xi[a] = p == traverser ? (legal[a] ? epsilon / n_legal : 0.0f) + (1.0f - epsilon) * sigma[a] : sigma[a];
    const int a = sample(xi, n, rng);
    history_observation(e, hist);
    value->forward(hist, b);
    Engine child = e;
    child.step(a);
    const float v_child = dream(child, own_reach * (p == traverser ? xi[a] : 1.0));
    float va[MAX_ACTIONS];
    for (int k = 0; k < n; ++k) va[k] = legal[k] ? b[k] : 0.0f;
    va[a] = b[a] + (v_child - b[a]) / std::max(xi[a], 1e-12f);
    float v = 0.0f;
    for (int k = 0; k < n; ++k) v += sigma[k] * va[k];
    if (p == traverser) {
      float target[MAX_ACTIONS];
      for (int k = 0; k < n; ++k) target[k] = legal[k] ? va[k] - v : 0.0f;
      adv.add(obs, float(t / std::max(own_reach, 1e-12)), target);
    }
    // expected-SARSA target for Q_traverser(h, a)
    float q_target;
    if (child.done) {
      q_target = float(child.rewards[traverser]);
    } else {
      float cobs[OBS_DIM], csig[MAX_ACTIONS], cb[MAX_ACTIONS], chist[OBS_DIM_WITH_OPP];
      bool clegal[MAX_ACTIONS];
      int ctwin[MAX_ACTIONS];
      sigma_at(child, cobs, csig, clegal, ctwin);
      history_observation(child, chist);
      value->forward(chist, cb);
      q_target = 0.0f;
      for (int k = 0; k < child.num_actions(); ++k) q_target += clegal[k] ? csig[k] * cb[k] : 0.0f;
    }
    float row[MAX_ACTIONS] = {};
    row[a] = q_target;
    val.add(hist, float(a), row);
    return v;
  }

  // -- ESCHER -----------------------------------------------------------------------------
  void escher_values(Engine& e) {  // one self-play trajectory under sigma: (h, a, u_0) rows
    std::vector<std::array<float, OBS_DIM_WITH_OPP>> hists;
    std::vector<int> acts;
    while (!e.done) {
      float obs[OBS_DIM], sigma[MAX_ACTIONS];
      bool legal[MAX_ACTIONS];
      int twin[MAX_ACTIONS];
      sigma_at(e, obs, sigma, legal, twin);
      ++nodes;
      std::array<float, OBS_DIM_WITH_OPP> h;
      history_observation(e, h.data());
      const int a = sample(sigma, e.num_actions(), rng);
      hists.push_back(h);
      acts.push_back(a);
      e.step(a);
    }
    const float u0 = float(e.rewards[0]);
    for (size_t i = 0; i < hists.size(); ++i) {
      float row[MAX_ACTIONS] = {};
      row[acts[i]] = u0;
      val.add(hists[i].data(), float(acts[i]), row);
    }
  }

  void escher_regrets(Engine& e) {  // update player = traverser samples uniformly, the opponent plays sigma
    while (!e.done) {
      const int p = e.current, n = e.num_actions();
      float obs[OBS_DIM], sigma[MAX_ACTIONS];
      bool legal[MAX_ACTIONS];
      int twin[MAX_ACTIONS];
      sigma_at(e, obs, sigma, legal, twin);
      ++nodes;
      int a;
      if (p == traverser) {
        float hist[OBS_DIM_WITH_OPP], q[MAX_ACTIONS], target[MAX_ACTIONS];
        history_observation(e, hist);
        value->forward(hist, q);
        const float sign = p == 0 ? 1.0f : -1.0f;
        float v = 0.0f;
        for (int k = 0; k < n; ++k) v += legal[k] ? sigma[k] * sign * q[k] : 0.0f;
        for (int k = 0; k < n; ++k) target[k] = legal[k] ? sign * q[k] - v : 0.0f;
        adv.add(obs, t, target);
        strat.add(obs, t, sigma);
        float row[MAX_ACTIONS] = {};
        row[0] = v;
        val.add(hist, -1.0f, row);  // the histories seen by the update player (tests / diagnostics)
        int cnt = 0;
        for (int k = 0; k < n; ++k) cnt += legal[k];
        std::uniform_int_distribution<int> d(0, cnt - 1);
        int pick = d(rng);
        a = 0;
        for (int k = 0; k < n; ++k)
          if (legal[k] && pick-- == 0) { a = k; break; }
      } else {
        a = sample(sigma, n, rng);
      }
      e.step(a);
    }
  }
};

py::tuple run_dream(std::shared_ptr<Model> net0, std::shared_ptr<Model> net1, std::shared_ptr<Model> baseline, int traverser,
                    int n_traversals, float t, float epsilon, uint64_t seed, EngineConfig cfg, py::object decks_obj) {
  if (!baseline->opp_cards) throw std::runtime_error("the DREAM baseline must be an opp_cards (history-input) network");
  TrajectorySampler ts;
  ts.nets[0] = net0.get(); ts.nets[1] = net1.get(); ts.value = baseline.get();
  ts.traverser = traverser; ts.t = t; ts.epsilon = epsilon;
  ts.rng.seed(seed);
  ts.adv.obs_dim = ts.strat.obs_dim = net0->obs_dim;
  ts.val.obs_dim = OBS_DIM_WITH_OPP;
  ts.adv.num_actions = ts.strat.num_actions = ts.val.num_actions = cfg.num_actions();
  std::vector<int> decks;  // optional fixed deals (n_traversals, >= 9)
  int stride = 0;
  if (!decks_obj.is_none()) {
    auto arr = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(decks_obj);
    if (!arr || arr.ndim() != 2 || arr.shape(0) != n_traversals || arr.shape(1) < 9)
      throw std::runtime_error("decks must be an int array of shape (n_traversals, >=9)");
    decks.assign(arr.data(), arr.data() + arr.size());
    stride = int(arr.shape(1));
  }
  {
    py::gil_scoped_release release;
    Engine e(cfg);
    for (int i = 0; i < n_traversals; ++i) {
      if (decks.empty()) e.reset_random(ts.rng);
      else e.reset(decks.data() + size_t(i) * stride);
      ts.dream(e, 1.0);
    }
  }
  const ssize_t na = ssize_t(ts.adv.size()), nv = ssize_t(ts.val.size()), nact = cfg.num_actions();
  return py::make_tuple(to_array(ts.adv.obs, na, ts.adv.obs_dim), to_array(ts.adv.t, na, 1), to_array(ts.adv.target, na, nact),
                        to_array(ts.val.obs, nv, OBS_DIM_WITH_OPP), to_array(ts.val.t, nv, 1), to_array(ts.val.target, nv, nact), ts.nodes);
}

py::tuple run_escher_values(std::shared_ptr<Model> net0, std::shared_ptr<Model> net1, int n_trajectories, uint64_t seed, EngineConfig cfg) {
  TrajectorySampler ts;
  ts.nets[0] = net0.get(); ts.nets[1] = net1.get();
  ts.rng.seed(seed);
  ts.val.obs_dim = OBS_DIM_WITH_OPP;
  ts.val.num_actions = cfg.num_actions();
  {
    py::gil_scoped_release release;
    Engine e(cfg);
    for (int i = 0; i < n_trajectories; ++i) {
      e.reset_random(ts.rng);
      ts.escher_values(e);
    }
  }
  const ssize_t nv = ssize_t(ts.val.size()), nact = cfg.num_actions();
  return py::make_tuple(to_array(ts.val.obs, nv, OBS_DIM_WITH_OPP), to_array(ts.val.t, nv, 1), to_array(ts.val.target, nv, nact), ts.nodes);
}

py::tuple run_escher_regrets(std::shared_ptr<Model> net0, std::shared_ptr<Model> net1, std::shared_ptr<Model> vnet, int traverser,
                             int n_trajectories, float t, uint64_t seed, EngineConfig cfg) {
  if (!vnet->opp_cards) throw std::runtime_error("the ESCHER value net must be an opp_cards (history-input) network");
  TrajectorySampler ts;
  ts.nets[0] = net0.get(); ts.nets[1] = net1.get(); ts.value = vnet.get();
  ts.traverser = traverser; ts.t = t;
  ts.rng.seed(seed);
  ts.adv.obs_dim = ts.strat.obs_dim = net0->obs_dim;
  ts.val.obs_dim = OBS_DIM_WITH_OPP;
  ts.adv.num_actions = ts.strat.num_actions = ts.val.num_actions = cfg.num_actions();
  {
    py::gil_scoped_release release;
    Engine e(cfg);
    for (int i = 0; i < n_trajectories; ++i) {
      e.reset_random(ts.rng);
      ts.escher_regrets(e);
    }
  }
  const ssize_t na = ssize_t(ts.adv.size()), ns = ssize_t(ts.strat.size()), nv = ssize_t(ts.val.size()), nact = cfg.num_actions();
  return py::make_tuple(to_array(ts.adv.obs, na, ts.adv.obs_dim), to_array(ts.adv.t, na, 1), to_array(ts.adv.target, na, nact),
                        to_array(ts.strat.obs, ns, ts.strat.obs_dim), to_array(ts.strat.t, ns, 1), to_array(ts.strat.target, ns, nact),
                        to_array(ts.val.obs, nv, OBS_DIM_WITH_OPP), to_array(ts.val.t, nv, 1), to_array(ts.val.target, nv, nact), ts.nodes);
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
        return e.cfg.has_all_in ? e.cfg.all_in() : e.cfg.num_actions() - 1;
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

struct ComboCards {  // inverse of combo_index
  uint8_t a[NUM_COMBOS], b[NUM_COMBOS];
  ComboCards() {
    for (int x = 0; x < NUM_CARDS; ++x)
      for (int y = x + 1; y < NUM_CARDS; ++y) { a[combo_index(x, y)] = uint8_t(x); b[combo_index(x, y)] = uint8_t(y); }
  }
};
static const ComboCards COMBO_CARDS;

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

// ------------------------------------------------------------------ depth-limited subgame search
// Unsafe subgame solving (Brown & Sandholm 2017) with the depth limit at the end of the current
// betting round, solved by external-sampling MCCFR with tabular regrets over (public sequence,
// hand) infosets and linear (LCFR) averaging; beyond the depth limit both players follow a
// blueprint continuation strategy (one of the given network pairs, sampled per rollout - Brown,
// Sandholm & Amos 2018 use several biased continuations, Pluribus rolls the blueprint out).  The
// root is a chance node dealing (hero, villain) hands from the given ranges (the blueprint's
// reach probabilities computed by the caller); unknown board cards are dealt per traversal.
struct Continuation {
  const Model* nets[2];
  bool rm;       // regret matching on advantage nets (true) or softmax on a policy net (false)
  double weight;
};

struct SubgameNode {
  Engine state;
  int kind = 0;   // 0 decision, 1 fold terminal, 2 showdown terminal, 3 street-end leaf (continue with the blueprint)
  int player = -1;
  int folder = -1, stake = 0;
  bool legal[MAX_ACTIONS] = {};
  int twin[MAX_ACTIONS] = {};
  int child[MAX_ACTIONS] = {};
};

struct SubgameSolver {
  std::vector<SubgameNode> nodes;
  int root_stage = 0, n_actions = 4, hero = 0;
  int known_board = 0;
  std::vector<double> range[2], cum[2];  // per seat: 1326 combo weights and cumulative sums
  std::vector<Continuation> conts;
  std::vector<double> cont_cum;
  std::vector<float> regret, strat_sum;  // (nodes x NUM_COMBOS x n_actions)
  std::vector<float> last_regret;        // PCFR+: the previous instantaneous regret (the prediction)
  std::vector<int> touched;              // per infoset: iteration of the last regret / strategy update (lazy discounting)
  std::vector<std::vector<std::shared_ptr<Model>>> keep_alive;
  std::mt19937_64 rng;
  int iteration = 0;
  // CFR variant: 0 LCFR (linear weights), 1 DCFR (Brown & Sandholm 2019: alpha 1.5, beta 0, gamma 2),
  // 2 CFR+ (regret floor, linear averaging), 3 PCFR+ (predictive RM+, quadratic averaging; Farina, Kroer & Sandholm 2021)
  int variant = 0;
  double alpha = 1.5, beta = 0.0, gamma = 2.0;
  std::vector<double> disc_pos, disc_neg, disc_strat;  // log prefix sums of the per-iteration DCFR discounts

  int add_node(const Engine& e) {
    SubgameNode nd;
    nd.state = e;
    nodes.push_back(nd);
    return int(nodes.size()) - 1;
  }

  void build(const Engine& root) {
    nodes.clear();
    root_stage = root.stage;
    n_actions = root.num_actions();
    known_board = BOARD_CARDS_BY_STAGE[root.stage];
    add_node(root);
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].kind != 0) continue;  // terminal / street-end leaf (kind set when the parent was expanded)
      Engine e = nodes[i].state;  // copy: nodes may reallocate
      SubgameNode& nd = nodes[i];
      nd.player = e.current;
      bool legal[MAX_ACTIONS];
      int twin[MAX_ACTIONS];
      e.legal_mask(legal, twin);
      for (int a = 0; a < n_actions; ++a) {
        nodes[i].legal[a] = legal[a];
        nodes[i].twin[a] = twin[a];
        nodes[i].child[a] = -1;
      }
      for (int a = 0; a < n_actions; ++a) {
        if (!legal[a]) continue;
        Engine c = e;
        c.step(a);
        const int id = add_node(c);
        nodes[i].child[a] = id;
        SubgameNode& ch = nodes[id];
        if (c.done) {
          if (c.folded >= 0) {
            ch.kind = 1;
            ch.folder = c.folded;
            ch.stake = c.bets[c.folded];
          } else {
            ch.kind = 2;
            ch.stake = std::min(c.bets[0], c.bets[1]);
          }
        } else if (c.stage != root_stage) {
          ch.kind = 3;
        }
      }
    }
    regret.assign(nodes.size() * NUM_COMBOS * n_actions, 0.0f);
    strat_sum.assign(nodes.size() * NUM_COMBOS * n_actions, 0.0f);
    last_regret.assign(nodes.size() * NUM_COMBOS * n_actions, 0.0f);
    touched.assign(nodes.size() * NUM_COMBOS, 0);
    iteration = 0;
  }

  void prepare_discounts(int total_iterations) {
    // DCFR: after iteration t, positive regrets x t^a/(t^a+1), negative x t^b/(t^b+1), strategy sums x (t/(t+1))^g;
    // stored as prefix products so an infoset untouched since iteration L can be discounted lazily
    // (in log space: t^a overflows and 0.5^t underflows long before a million iterations)
    const int n = total_iterations + 2;
    disc_pos.assign(n, 0.0);
    disc_neg.assign(n, 0.0);
    disc_strat.assign(n, 0.0);
    for (int t = 1; t < n; ++t) {
      const double lt = std::log(double(t));
      disc_pos[t] = disc_pos[t - 1] - std::log1p(std::exp(-alpha * lt));   // log(t^a / (t^a + 1))
      disc_neg[t] = disc_neg[t - 1] - std::log1p(std::exp(-beta * lt));
      disc_strat[t] = disc_strat[t - 1] + gamma * (lt - std::log(double(t) + 1.0));
    }
  }

  // bring an infoset's accumulators up to date before adding iteration `iteration`'s contribution
  void catch_up(size_t info) {
    if (variant != 1) return;
    const int last = touched[info];
    if (last == iteration || last == 0 && iteration == 1) {
      touched[info] = iteration;
      return;
    }
    // discounts for the finished iterations last .. iteration-1
    const int from = std::max(last - 1, 0);
    const double fp = std::exp(disc_pos[iteration - 1] - disc_pos[from]);
    const double fn = std::exp(disc_neg[iteration - 1] - disc_neg[from]);
    const double fs = std::exp(disc_strat[iteration - 1] - disc_strat[from]);
    float* R = regret.data() + info * n_actions;
    float* S = strat_sum.data() + info * n_actions;
    for (int a = 0; a < n_actions; ++a) {
      R[a] = float(R[a] > 0 ? R[a] * fp : R[a] * fn);
      S[a] = float(S[a] * fs);
    }
    touched[info] = iteration;
  }

  float strategy_weight() const {  // weight of this iteration's strategy contribution
    switch (variant) {
      case 0: return float(iteration);                            // LCFR: linear
      case 1: return 1.0f;                                        // DCFR: (discounted afterwards by (t/(t+1))^gamma)
      case 2: return float(iteration);                            // CFR+: linear averaging
      default: return float(iteration) * float(iteration);        // PCFR+: quadratic averaging
    }
  }

  void set_ranges(const float* r0, const float* r1) {
    for (int s = 0; s < 2; ++s) {
      const float* r = s == 0 ? r0 : r1;
      range[s].assign(r, r + NUM_COMBOS);
      cum[s].resize(NUM_COMBOS);
      double acc = 0.0;
      for (int h = 0; h < NUM_COMBOS; ++h) {
        acc += std::max(0.0, double(r[h]));
        cum[s][h] = acc;
      }
      if (acc <= 0) throw std::runtime_error("empty range");
    }
  }

  int sample_hand(int seat) {
    std::uniform_real_distribution<double> u(0.0, cum[seat].back());
    const double x = u(rng);
    return int(std::lower_bound(cum[seat].begin(), cum[seat].end(), x) - cum[seat].begin());
  }

  static void combo_cards(int h, int& a, int& b) {  // inverse of combo_index (a < b), tabulated
    a = COMBO_CARDS.a[h];
    b = COMBO_CARDS.b[h];
  }

  // -- one traversal ----------------------------------------------------------------
  int hand[2][2];
  int board[5];
  int traverser;
  float t_weight;  // linear CFR iteration weight

  const float* rm_sigma(int node, int seat, int h, float* sigma) {
    const size_t info = size_t(node) * NUM_COMBOS + h;
    const float* R = regret.data() + info * n_actions;
    const SubgameNode& nd = nodes[node];
    if (variant == 3) {  // predictive: regret matching on R + prediction (the last instantaneous regret)
      float pred[MAX_ACTIONS];
      const float* L = last_regret.data() + info * n_actions;
      for (int a = 0; a < n_actions; ++a) pred[a] = R[a] + L[a];
      regret_matching(pred, sigma, nd.legal, n_actions, false);
    } else {
      regret_matching(R, sigma, nd.legal, n_actions, false);
    }
    (void)seat;
    return sigma;
  }

  void update_regrets(size_t info, const float* inst) {  // inst[a] = v_a - v (already for legal actions)
    catch_up(info);
    float* R = regret.data() + info * n_actions;
    switch (variant) {
      case 0:
        for (int a = 0; a < n_actions; ++a) R[a] += t_weight * inst[a];
        break;
      case 1:
        for (int a = 0; a < n_actions; ++a) R[a] += inst[a];
        break;
      default: {  // CFR+ / PCFR+: regret matching plus (floor at zero)
        float* L = last_regret.data() + info * n_actions;
        for (int a = 0; a < n_actions; ++a) {
          R[a] = std::max(0.0f, R[a] + inst[a]);
          L[a] = inst[a];
        }
      }
    }
  }

  float rollout(const SubgameNode& nd) {
    // continue from the street-end leaf with a sampled continuation strategy until the hand ends
    Engine e = nd.state;
    e.hands[0][0] = hand[0][0]; e.hands[0][1] = hand[0][1];
    e.hands[1][0] = hand[1][0]; e.hands[1][1] = hand[1][1];
    for (int i = 0; i < 5; ++i) e.board[i] = board[i];
    std::uniform_real_distribution<double> u(0.0, cont_cum.back());
    const Continuation& c = conts[size_t(std::lower_bound(cont_cum.begin(), cont_cum.end(), u(rng)) - cont_cum.begin())];
    float obs[OBS_DIM], out[MAX_ACTIONS], sigma[MAX_ACTIONS];
    bool legal[MAX_ACTIONS];
    while (!e.done) {
      const int p = e.current;
      e.observation(-1, obs);
      c.nets[p]->forward(obs, out);
      e.legal_mask(legal);
      if (c.rm) {
        regret_matching(out, sigma, legal, n_actions, c.nets[p]->rm_argmax);
      } else {
        for (int a = 0; a < n_actions; ++a)
          if (!legal[a]) out[a] = -1e30f;
        softmax(out, sigma, n_actions);
      }
      e.step(sample(sigma, n_actions, rng));
    }
    return float(e.rewards[traverser]);
  }

  float terminal_value(const SubgameNode& nd) {
    if (nd.kind == 1) return nd.folder == traverser ? -float(nd.stake) : float(nd.stake);
    // showdown on the (sampled) board of the game's last round
    int c0[7] = {hand[0][0], hand[0][1], board[0], board[1], board[2], board[3], board[4]};
    int c1[7] = {hand[1][0], hand[1][1], board[0], board[1], board[2], board[3], board[4]};
    const int nb = nodes[0].state.cfg.showdown_cards();
    const int s0 = eval_best(c0, 2 + nb), s1 = eval_best(c1, 2 + nb);
    if (s0 == s1) return 0.0f;
    const int winner = s0 < s1 ? 0 : 1;
    return winner == traverser ? float(nd.stake) : -float(nd.stake);
  }

  float traverse(int node_id) {
    const SubgameNode& nd = nodes[node_id];
    if (nd.kind == 1 || nd.kind == 2) return terminal_value(nd);
    if (nd.kind == 3) return rollout(nd);
    const int p = nd.player;
    const int h = combo_index(std::min(hand[p][0], hand[p][1]), std::max(hand[p][0], hand[p][1]));
    float sigma[MAX_ACTIONS];
    rm_sigma(node_id, p, h, sigma);
    const size_t base = (size_t(node_id) * NUM_COMBOS + h) * n_actions;
    if (p == traverser) {
      float va[MAX_ACTIONS];
      for (int a = 0; a < n_actions; ++a)
        if (nd.legal[a]) va[a] = traverse(nd.child[a]);
      for (int a = 0; a < n_actions; ++a)
        if (!nd.legal[a]) va[a] = va[nd.twin[a]];
      float value = 0.0f;
      for (int a = 0; a < n_actions; ++a) value += sigma[a] * va[a];
      float inst[MAX_ACTIONS];
      for (int a = 0; a < n_actions; ++a) inst[a] = nd.legal[a] ? va[a] - value : 0.0f;
      update_regrets(base / n_actions, inst);
      return value;
    }
    catch_up(base / n_actions);
    const float sw = strategy_weight();
    for (int a = 0; a < n_actions; ++a) strat_sum[base + a] += sw * sigma[a];
    return traverse(nd.child[sample(sigma, n_actions, rng)]);
  }

  // ``hero`` / ``hero_hand``: the seat we are solving for and its real hand (combo index).  On the
  // hero's own traversals the hero is dealt its real hand with probability ``focus`` instead of a
  // range sample ("targeted" sampling): regret matching is per infoset and scale-free, so this
  // only concentrates the hero's updates on the infosets it will actually play; the villain's
  // traversals always deal the hero from its range, so the villain still answers the whole range
  // and cannot exploit knowledge of the real hand.
  void run(int iterations, uint64_t seed, int hero_seat = 0, int hero_hand = -1, double focus = 0.0) {
    rng.seed(seed);
    hero = hero_seat;
    if (variant == 1) prepare_discounts(iteration + iterations);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int it = 0; it < iterations; ++it) {
      ++iteration;
      t_weight = float(iteration);  // linear CFR
      for (int trav = 0; trav < 2; ++trav) {
        // deal: hero / villain hands from the ranges (rejecting overlaps), the unknown board cards uniformly
        for (int tries = 0;; ++tries) {
          const bool focused = trav == hero && hero_hand >= 0 && focus > 0.0 && u01(rng) < focus;
          const int hh = focused ? hero_hand : sample_hand(hero);
          const int hv = sample_hand(1 - hero);
          const int h0 = hero == 0 ? hh : hv, h1 = hero == 0 ? hv : hh;
          int a0, b0, a1, b1;
          combo_cards(h0, a0, b0);
          combo_cards(h1, a1, b1);
          if (a0 != a1 && a0 != b1 && b0 != a1 && b0 != b1) {
            hand[0][0] = a0; hand[0][1] = b0; hand[1][0] = a1; hand[1][1] = b1;
            break;
          }
          if (tries > 10000) throw std::runtime_error("ranges only overlap");
        }
        bool used[NUM_CARDS] = {};
        for (int s = 0; s < 2; ++s) { used[hand[s][0]] = true; used[hand[s][1]] = true; }
        const Engine& root = nodes[0].state;
        for (int i = 0; i < known_board; ++i) { board[i] = root.board[i]; used[board[i]] = true; }
        for (int i = known_board; i < 5; ++i) {  // deal the rest (only the last round's cards matter)
          std::uniform_int_distribution<int> d(0, NUM_CARDS - 1);
          int c;
          do { c = d(rng); } while (used[c]);
          used[c] = true;
          board[i] = c;
        }
        traverser = trav;
        traverse(0);
      }
    }
  }

  // Warm start: copy the regret / strategy tables of the subtree rooted at `old_root` of a previous
  // solve of the same street (nested re-solving: the new root is a node of the previous tree; the
  // trees under both roots are identical up to node numbering).
  // Regrets are copied scaled to `weight` equivalent iterations (Brown & Sandholm 2016, strategy-based
  // warm starting: the copied state behaves like a short run that already reached the old strategy),
  // the average-strategy sums are reset (they were accumulated under the pre-action ranges).
  void warm_start(const SubgameSolver& prev, int old_root, int weight = 2000) {
    if (prev.n_actions != n_actions) throw std::runtime_error("warm start: different games");
    if (prev.iteration <= 0) return;
    const double scale = double(weight) / double(prev.iteration);
    std::vector<std::pair<int, int>> stack = {{0, old_root}};
    while (!stack.empty()) {
      auto [mine, theirs] = stack.back();
      stack.pop_back();
      const SubgameNode& a = nodes[mine];
      const SubgameNode& b = prev.nodes[theirs];
      if (a.kind != b.kind || a.player != b.player) throw std::runtime_error("warm start: subtree mismatch");
      if (a.kind == 0) {
        const size_t na = size_t(mine) * NUM_COMBOS * n_actions, nb = size_t(theirs) * NUM_COMBOS * n_actions;
        for (size_t i = 0; i < size_t(NUM_COMBOS) * n_actions; ++i) regret[na + i] = float(prev.regret[nb + i] * scale);
        for (int act = 0; act < n_actions; ++act)
          if (a.legal[act] && b.legal[act]) stack.push_back({a.child[act], b.child[act]});
      }
    }
    iteration = weight;
  }

  // average strategy of every hand at a decision node: (NUM_COMBOS, n_actions)
  py::array_t<float> node_strategy(int node_id) {
    if (node_id < 0 || node_id >= int(nodes.size()) || nodes[node_id].kind != 0) throw std::runtime_error("not a decision node");
    py::array_t<float> out({ssize_t(NUM_COMBOS), ssize_t(n_actions)});
    auto o = out.mutable_unchecked<2>();
    const SubgameNode& nd = nodes[node_id];
    for (int h = 0; h < NUM_COMBOS; ++h) {
      const size_t base = (size_t(node_id) * NUM_COMBOS + h) * n_actions;
      double total = 0.0;
      for (int a = 0; a < n_actions; ++a) total += strat_sum[base + a];
      if (total > 0) {
        for (int a = 0; a < n_actions; ++a) o(h, a) = float(strat_sum[base + a] / total);
      } else {  // never visited as the opponent: current regret-matching strategy
        float sigma[MAX_ACTIONS];
        regret_matching(regret.data() + base, sigma, nd.legal, n_actions, false);
        for (int a = 0; a < n_actions; ++a) o(h, a) = sigma[a];
      }
    }
    return out;
  }
};

// ------------------------------------------------------------------ public betting tree of a (sub)game
struct PublicTree {
  std::vector<SubgameNode> nodes;
  std::vector<int> round;

  void build(const Engine& root) {
    nodes.clear();
    round.clear();
    const int n_actions = root.num_actions();
    SubgameNode rn;
    rn.state = root;
    nodes.push_back(rn);
    round.push_back(root.stage);
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].kind != 0) continue;
      Engine e = nodes[i].state;
      nodes[i].player = e.current;
      bool legal[MAX_ACTIONS];
      int twin[MAX_ACTIONS];
      e.legal_mask(legal, twin);
      for (int a = 0; a < n_actions; ++a) {
        nodes[i].legal[a] = legal[a];
        nodes[i].twin[a] = twin[a];
        nodes[i].child[a] = -1;
      }
      for (int a = 0; a < n_actions; ++a) {
        if (!legal[a]) continue;
        Engine c = e;
        c.step(a);
        SubgameNode ch;
        ch.state = c;
        if (c.done) {
          if (c.folded >= 0) {
            ch.kind = 1;
            ch.folder = c.folded;
            ch.stake = c.bets[c.folded];
          } else {
            ch.kind = 2;
            ch.stake = std::min(c.bets[0], c.bets[1]);
          }
        }
        nodes[i].child[a] = int(nodes.size());
        nodes.push_back(ch);
        round.push_back(c.stage);
      }
    }
  }
};

// ------------------------------------------------------------------ vector-form public-chance-sampling solver
// Solves the rest of the hand from the start of a betting round (Pluribus's heads-up search: from
// the flop on the subgame extends to the end of the game).  Vector form: every iteration processes
// all 1326 hands of both players at every public node (reach vectors in, counterfactual value
// vectors out); terminal values in O(1326 + 52 * 51) per node via strength-sorted cumulative sums
// with blocker corrections.  Chance is *public-chance sampled*: each iteration deals the unknown
// board cards once; hands blocked by a newly dealt card leave the reach vectors at that street
// transition and only hands compatible with the whole sampled board are updated, which makes every
// hand's estimate unbiased (its updates average over the boards it is compatible with).  Infosets
// of the root round are lossless (per hand); later rounds use `buckets` equity buckets per round
// (Pluribus: lossless in the current round, 500 lossy buckets on later rounds; ours are equal-width
// buckets of the hand's equity against a uniform range, not k-means over equity distributions).
// Linear CFR (Pluribus) by default, DCFR / CFR+ / PCFR+ single-threaded.  Threads share the
// tables, one sampled board per thread and iteration (racy but benign updates, as in Pluribus).
// A hand's actions already taken in the round can be frozen (Pluribus's nested unsafe search keeps
// the searcher's real hand on the actions it has taken; other hands are free).
struct BoardTable {  // strengths and strength-sorted combo orders of one complete board
  std::vector<int> strength;                 // per combo, INT32_MAX for combos overlapping the board
  std::vector<int> order;                    // valid combos, strongest first
  std::vector<std::vector<int>> card_order;  // per card: its valid combos, strongest first
  std::vector<float> equity;                 // per combo: P(win) + P(tie) / 2 vs a uniform compatible hand (-1 if blocked)

  void build(const int* board, int nb) {
    strength.assign(NUM_COMBOS, INT32_MAX);
    bool onboard[NUM_CARDS] = {};
    for (int i = 0; i < nb; ++i) onboard[board[i]] = true;
    order.clear();
    card_order.assign(NUM_CARDS, {});
    for (int a = 0; a < NUM_CARDS; ++a)
      for (int b = a + 1; b < NUM_CARDS; ++b) {
        if (onboard[a] || onboard[b]) continue;
        int c7[7] = {a, b, board[0], board[1], board[2], board[3], board[4]};
        const int h = combo_index(a, b);
        strength[h] = eval_best(c7, 2 + nb);
        order.push_back(h);
        card_order[a].push_back(h);
        card_order[b].push_back(h);
      }
    auto by_strength = [&](int x, int y) { return strength[x] < strength[y]; };
    std::stable_sort(order.begin(), order.end(), by_strength);
    for (auto& v : card_order) std::stable_sort(v.begin(), v.end(), by_strength);
    // equity of every valid combo against the uniform range of compatible combos
    std::vector<double> ones(NUM_COMBOS, 0.0), mass(NUM_COMBOS), sd(NUM_COMBOS);
    for (int h : order) ones[h] = 1.0;
    opponent_mass(ones, mass);
    showdown_values(ones, sd);
    equity.assign(NUM_COMBOS, -1.0f);
    for (int h : order) equity[h] = mass[h] > 0 ? float(0.5 + 0.5 * sd[h] / mass[h]) : 0.5f;
  }

  // M[h] = sum of reach over combos compatible with h (any board: the reach carries the validity)
  static void opponent_mass(const std::vector<double>& reach, std::vector<double>& out) {
    double total = 0.0, per_card[NUM_CARDS] = {};
    for (int h = 0; h < NUM_COMBOS; ++h) {
      const double r = reach[h];
      if (r == 0.0) continue;
      total += r;
      int a, b;
      SubgameSolver::combo_cards(h, a, b);
      per_card[a] += r;
      per_card[b] += r;
    }
    for (int h = 0; h < NUM_COMBOS; ++h) {
      int a, b;
      SubgameSolver::combo_cards(h, a, b);
      out[h] = total - per_card[a] - per_card[b] + reach[h];
    }
  }

  // W[h] - L[h]: reach mass of compatible weaker combos minus stronger ones (ties count 0)
  void showdown_values(const std::vector<double>& reach, std::vector<double>& out) const {
    for (int h = 0; h < NUM_COMBOS; ++h) out[h] = 0.0;
    const size_t n = order.size();
    thread_local std::vector<double> stronger, weaker;
    stronger.assign(NUM_COMBOS, 0.0);
    weaker.assign(NUM_COMBOS, 0.0);
    double total = 0.0;
    for (int h : order) total += reach[h];
    double cum = 0.0;
    for (size_t i = 0; i < n;) {
      size_t j = i;
      double group = 0.0;
      while (j < n && strength[order[j]] == strength[order[i]]) group += reach[order[j++]];
      for (size_t k = i; k < j; ++k) {
        stronger[order[k]] = cum;
        weaker[order[k]] = total - cum - group;
      }
      cum += group;
      i = j;
    }
    for (int c = 0; c < NUM_CARDS; ++c) {  // combos sharing a card cannot be held against h
      const auto& v = card_order[c];
      const size_t m = v.size();
      if (!m) continue;
      double tot = 0.0;
      for (int h : v) tot += reach[h];
      double cs = 0.0;
      for (size_t i = 0; i < m;) {
        size_t j = i;
        double group = 0.0;
        while (j < m && strength[v[j]] == strength[v[i]]) group += reach[v[j++]];
        for (size_t k = i; k < j; ++k) {
          stronger[v[k]] -= cs;
          weaker[v[k]] -= tot - cs - group;
        }
        cs += group;
        i = j;
      }
    }
    for (int h : order) out[h] = weaker[h] - stronger[h];
  }
};

struct VectorSolver {
  std::vector<SubgameNode> nodes;
  std::vector<int> node_round, info_offset;
  int root_round = 0, last_round = 0, n_actions = 4, buckets = 500;
  int known_board = 0, n_unknown = 0, showdown_cards = 5;
  int root_board[5] = {0, 0, 0, 0, 0};
  std::vector<int> deck;  // cards not on the known board
  std::vector<double> range[2];
  std::vector<double> regret, strat_sum, last_regret;  // per infoset x action
  size_t n_infosets = 0;
  int variant = 0;  // 0 LCFR, 1 DCFR, 2 CFR+, 3 PCFR+ (see the river solver)
  double alpha = 1.5, beta = 0.0, gamma = 2.0;
  std::atomic<int> iteration{0};
  int last_seed_threads = 1;
  // tables over the sampled cards: complete boards (all n_unknown cards, as a set) and, per later
  // round, the buckets of every hand given the cards dealt so far (ordered tuple codes, base 52)
  std::unordered_map<int, BoardTable> boards;                                  // code(sorted unknown cards)
  std::vector<std::unordered_map<int, std::vector<uint16_t>>> round_buckets;  // [round]: code(prefix) -> bucket per hand
  struct Frozen { int node, hand, action; };
  std::vector<Frozen> frozen;
  std::vector<char> node_frozen;

  static int code_of(const int* cards, int n) {
    int c = 0;
    for (int i = 0; i < n; ++i) c = c * NUM_CARDS + cards[i];
    return c;
  }
  static int sorted_code(const int* cards, int n) {
    int tmp[5];
    for (int i = 0; i < n; ++i) tmp[i] = cards[i];
    std::sort(tmp, tmp + n);
    return code_of(tmp, n);
  }
  int n_keys(int node) const { return node_round[node] == root_round ? NUM_COMBOS : buckets; }

  void build(const Engine& root) {
    if (root.done) throw std::runtime_error("VectorSolver: the hand is over");
    root_round = root.stage;
    last_round = root.cfg.num_rounds - 1;
    showdown_cards = root.cfg.showdown_cards();
    known_board = BOARD_CARDS_BY_STAGE[root_round];
    n_unknown = showdown_cards - known_board;
    if (n_unknown > 2) throw std::runtime_error("VectorSolver solves from the flop on (at most two unknown board cards)");
    n_actions = root.num_actions();
    for (int i = 0; i < 5; ++i) root_board[i] = root.board[i];
    // public tree of the whole remaining hand
    PublicTree pt;
    pt.build(root);
    nodes = std::move(pt.nodes);
    node_round = std::move(pt.round);
    info_offset.assign(nodes.size(), 0);
    n_infosets = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
      info_offset[i] = int(n_infosets);
      if (nodes[i].kind == 0) n_infosets += size_t(n_keys(int(i)));
    }
    regret.assign(n_infosets * n_actions, 0.0);
    strat_sum.assign(n_infosets * n_actions, 0.0);
    last_regret.assign(n_infosets * n_actions, 0.0);
    frozen.clear();
    node_frozen.assign(nodes.size(), 0);
    iteration = 0;
    // deck and tables
    deck.clear();
    bool onboard[NUM_CARDS] = {};
    for (int i = 0; i < known_board; ++i) onboard[root_board[i]] = true;
    for (int c = 0; c < NUM_CARDS; ++c)
      if (!onboard[c]) deck.push_back(c);
    build_tables();
  }

  void build_tables() {
    boards.clear();
    round_buckets.assign(last_round + 1, {});
    // every set of unknown cards -> complete board table (threads over the sets)
    std::vector<std::vector<int>> sets;
    if (n_unknown == 0) sets.push_back({});
    else if (n_unknown == 1)
      for (int c : deck) sets.push_back({c});
    else
      for (size_t i = 0; i < deck.size(); ++i)
        for (size_t j = i + 1; j < deck.size(); ++j) sets.push_back({deck[i], deck[j]});
    std::vector<BoardTable> tabs(sets.size());
    auto work = [&](size_t lo, size_t hi) {
      for (size_t k = lo; k < hi; ++k) {
        int board[5];
        for (int i = 0; i < 5; ++i) board[i] = root_board[i];
        for (int i = 0; i < n_unknown; ++i) board[known_board + i] = sets[k][i];
        tabs[k].build(board, showdown_cards);
      }
    };
    const size_t nt = std::min<size_t>(sets.size(), std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
    std::vector<std::thread> pool;
    for (size_t t = 0; t < nt; ++t)
      pool.emplace_back(work, sets.size() * t / nt, sets.size() * (t + 1) / nt);
    for (auto& th : pool) th.join();
    for (size_t k = 0; k < sets.size(); ++k) boards[code_of(sets[k].data(), n_unknown)] = std::move(tabs[k]);
    // buckets: last round = equity on the complete board; the round before (only when two cards
    // are unknown) = mean equity over the completions compatible with the hand
    if (n_unknown >= 1) {
      auto& last = round_buckets[last_round];
      for (auto& kv : boards) {  // keyed by the sorted code; ordered tuples are mapped in bucket_table()
        std::vector<uint16_t> b(NUM_COMBOS, 0);
        for (int h = 0; h < NUM_COMBOS; ++h) {
          const float e = kv.second.equity[h];
          b[h] = e < 0 ? 0 : uint16_t(std::min(buckets - 1, int(e * buckets)));
        }
        last[kv.first] = std::move(b);
      }
    }
    if (n_unknown == 2) {
      auto& mid = round_buckets[last_round - 1];
      for (int c : deck) {
        std::vector<double> acc(NUM_COMBOS, 0.0), cnt(NUM_COMBOS, 0.0);
        for (int r : deck) {
          if (r == c) continue;
          int pair[2] = {c, r};
          const BoardTable& bt = boards.at(sorted_code(pair, 2));
          for (int h : bt.order) { acc[h] += bt.equity[h]; cnt[h] += 1.0; }
        }
        std::vector<uint16_t> b(NUM_COMBOS, 0);
        for (int h = 0; h < NUM_COMBOS; ++h)
          if (cnt[h] > 0) b[h] = uint16_t(std::min(buckets - 1, int(acc[h] / cnt[h] * buckets)));
        mid[c] = std::move(b);
      }
    }
  }

  // bucket table of a round given the unknown cards dealt so far (ordered as dealt)
  const std::vector<uint16_t>& bucket_table(int round, const int* drawn) const {
    const int n = BOARD_CARDS_BY_STAGE[round] - known_board;  // cards dealt since the root
    if (round == last_round) return round_buckets[round].at(sorted_code(drawn, n));
    return round_buckets[round].at(code_of(drawn, n));
  }

  void set_ranges(const float* r0, const float* r1) {
    bool onboard[NUM_CARDS] = {};
    for (int i = 0; i < known_board; ++i) onboard[root_board[i]] = true;
    for (int s = 0; s < 2; ++s) {
      const float* r = s == 0 ? r0 : r1;
      range[s].assign(NUM_COMBOS, 0.0);
      for (int h = 0; h < NUM_COMBOS; ++h) {
        int a, b;
        SubgameSolver::combo_cards(h, a, b);
        range[s][h] = (onboard[a] || onboard[b]) ? 0.0 : std::max(0.0, double(r[h]));
      }
    }
  }

  void freeze(int node, int hand, int action) {
    if (node < 0 || node >= int(nodes.size()) || nodes[node].kind != 0) throw std::runtime_error("freeze: not a decision node");
    if (node_round[node] != root_round) throw std::runtime_error("freeze: only root-round nodes (per-hand infosets)");
    if (!nodes[node].legal[action]) throw std::runtime_error("freeze: illegal action");
    frozen.push_back({node, hand, action});
    node_frozen[node] = 1;
  }

  void sigma_of(size_t info, const bool* legal, double* sigma) const {
    const double* R = regret.data() + info * n_actions;
    double pos[MAX_ACTIONS], total = 0.0;
    int cnt = 0;
    for (int a = 0; a < n_actions; ++a) {
      double r = R[a];
      if (variant == 3) r += last_regret[info * n_actions + a];
      pos[a] = (legal[a] && r > 0) ? r : 0.0;
      total += pos[a];
      cnt += legal[a];
    }
    for (int a = 0; a < n_actions; ++a) sigma[a] = total > 1e-12 ? pos[a] / total : (legal[a] ? 1.0 / cnt : 0.0);
  }

  // per-thread traversal state
  struct Pass {
    int drawn[2] = {-1, -1};
    int board[5];
    const BoardTable* table = nullptr;
    std::vector<char> valid;  // hand compatible with all drawn cards
    double t = 1.0;
    struct Level { std::vector<double> next, child, sigma, ua; std::vector<int> key; };
    std::deque<Level> levels;  // deque: growing it keeps the references held by the callers valid
    Level& level(size_t d) {
      while (d >= levels.size()) levels.emplace_back();
      Level& L = levels[d];
      if (L.next.empty()) {
        L.next.assign(NUM_COMBOS, 0.0);
        L.child.assign(NUM_COMBOS, 0.0);
        L.sigma.assign(size_t(NUM_COMBOS) * MAX_ACTIONS, 0.0);
        L.ua.assign(size_t(NUM_COMBOS) * MAX_ACTIONS, 0.0);
        L.key.assign(NUM_COMBOS, 0);
      }
      return L;
    }
  };

  void keys_of(int node, const Pass& ps, std::vector<int>& key) const {
    if (node_round[node] == root_round) {
      for (int h = 0; h < NUM_COMBOS; ++h) key[h] = h;
      return;
    }
    const std::vector<uint16_t>& b = bucket_table(node_round[node], ps.drawn);
    for (int h = 0; h < NUM_COMBOS; ++h) key[h] = b[h];
  }

  // counterfactual values of player p's hands at `node` (per hand, all 1326)
  void values(int node, int p, const std::vector<double>& reach_p, const std::vector<double>& reach_q,
              std::vector<double>& out, bool update, Pass& ps, size_t depth) {
    const SubgameNode& nd = nodes[node];
    bool any = false;  // nobody gets here: nothing to evaluate or accumulate (exact pruning; skipping on the
    for (int h = 0; h < NUM_COMBOS && !any; ++h) any = reach_q[h] != 0.0 || reach_p[h] != 0.0;  // opponent's reach
    if (!any) {                                                                                  // alone would bias the averages)
      for (int h = 0; h < NUM_COMBOS; ++h) out[h] = 0.0;
      return;
    }
    if (nd.kind == 1) {
      BoardTable::opponent_mass(reach_q, out);
      const double sign = nd.folder == p ? -1.0 : 1.0;
      for (int h = 0; h < NUM_COMBOS; ++h) out[h] *= sign * nd.stake;
      return;
    }
    if (nd.kind == 2) {
      ps.table->showdown_values(reach_q, out);
      for (int h = 0; h < NUM_COMBOS; ++h) out[h] *= nd.stake;
      return;
    }
    Pass::Level& L = ps.level(depth);
    keys_of(node, ps, L.key);
    const int A = n_actions, nk = n_keys(node);
    const size_t base = size_t(info_offset[node]);
    double* SK = L.sigma.data();  // strategy per infoset key (nk x A); hands read it through their key
    for (int k = 0; k < nk; ++k) sigma_of(base + k, nd.legal, SK + size_t(k) * A);
    if (node_frozen[node])  // root round only: key == hand
      for (const Frozen& f : frozen)
        if (f.node == node)
          for (int a = 0; a < A; ++a) SK[size_t(f.hand) * A + a] = a == f.action ? 1.0 : 0.0;
    for (int h = 0; h < NUM_COMBOS; ++h) out[h] = 0.0;
    const int round = node_round[node];
    const int* key = L.key.data();
    auto descend = [&](int a, const std::vector<double>& rp, const std::vector<double>& rq, std::vector<double>& res) {
      const int c = nd.child[a];
      if (nodes[c].kind == 0 && node_round[c] != round) {
        // street transition: the newly dealt card(s) block some hands of both players
        const int lo = BOARD_CARDS_BY_STAGE[round], hi = BOARD_CARDS_BY_STAGE[node_round[c]];
        std::vector<double> mp(rp), mq(rq);  // masked copies (a handful of transitions per iteration)
        for (int h = 0; h < NUM_COMBOS; ++h) {
          int x, y;
          SubgameSolver::combo_cards(h, x, y);
          bool blocked = false;
          for (int i = lo; i < hi; ++i) blocked |= (ps.board[i] == x || ps.board[i] == y);
          if (blocked) { mp[h] = 0.0; mq[h] = 0.0; }
        }
        values(c, p, mp, mq, res, update, ps, depth + 1);
      } else {
        values(c, p, rp, rq, res, update, ps, depth + 1);
      }
    };
    if (nd.player == p) {
      for (int a = 0; a < A; ++a) {
        if (!nd.legal[a]) continue;
        for (int h = 0; h < NUM_COMBOS; ++h) L.next[h] = reach_p[h] * SK[size_t(key[h]) * A + a];
        descend(a, L.next, reach_q, L.child);
        double* U = L.ua.data() + size_t(a) * NUM_COMBOS;
        for (int h = 0; h < NUM_COMBOS; ++h) {
          U[h] = L.child[h];
          out[h] += SK[size_t(key[h]) * A + a] * L.child[h];
        }
      }
      if (update) {
        const double t = ps.t;
        const double sw = variant == 0 ? t : variant == 1 ? 1.0 : variant == 2 ? t : t * t;
        for (int h = 0; h < NUM_COMBOS; ++h) {
          if (!ps.valid[h] || range[p][h] <= 0) continue;
          const size_t info = base + key[h];
          double* R = regret.data() + info * A;
          double* S = strat_sum.data() + info * A;
          const double* sk = SK + size_t(key[h]) * A;
          for (int a = 0; a < A; ++a) {
            if (!nd.legal[a]) continue;
            const double inst = L.ua[size_t(a) * NUM_COMBOS + h] - out[h];
            switch (variant) {
              case 0: R[a] += t * inst; break;
              case 1: R[a] += inst; break;
              default: R[a] = std::max(0.0, R[a] + inst); last_regret[info * A + a] = inst; break;
            }
            S[a] += sw * reach_p[h] * sk[a];
          }
        }
      }
      return;
    }
    for (int a = 0; a < A; ++a) {
      if (!nd.legal[a]) continue;
      for (int h = 0; h < NUM_COMBOS; ++h) L.next[h] = reach_q[h] * SK[size_t(key[h]) * A + a];
      descend(a, reach_p, L.next, L.child);
      for (int h = 0; h < NUM_COMBOS; ++h) out[h] += L.child[h];
    }
  }

  void discount() {  // DCFR: after each iteration (single-threaded variant)
    if (variant != 1) return;
    const double t = double(iteration.load());
    const double ta = std::pow(t, alpha), tb = std::pow(t, beta);
    const double fp = ta / (ta + 1.0), fn = tb / (tb + 1.0), fs = std::pow(t / (t + 1.0), gamma);
    for (size_t i = 0; i < regret.size(); ++i) {
      regret[i] *= regret[i] > 0 ? fp : fn;
      strat_sum[i] *= fs;
    }
  }

  void one_iteration(Pass& ps, std::mt19937_64& rng) {
    // deal the unknown cards (partial Fisher-Yates over a copy of the deck)
    std::vector<int> pool(deck);
    for (int i = 0; i < n_unknown; ++i) {
      std::uniform_int_distribution<int> d(i, int(pool.size()) - 1);
      std::swap(pool[i], pool[d(rng)]);
      ps.drawn[i] = pool[i];
    }
    for (int i = 0; i < 5; ++i) ps.board[i] = root_board[i];
    for (int i = 0; i < n_unknown; ++i) ps.board[known_board + i] = ps.drawn[i];
    ps.table = &boards.at(sorted_code(ps.drawn, n_unknown));
    ps.valid.assign(NUM_COMBOS, 1);
    for (int i = 0; i < n_unknown; ++i)
      for (int c = 0; c < NUM_CARDS; ++c) {
        if (c == ps.drawn[i]) continue;
        const int lo = std::min(c, ps.drawn[i]), hi = std::max(c, ps.drawn[i]);
        ps.valid[combo_index(lo, hi)] = 0;
      }
    ps.t = double(++iteration);
    std::vector<double> out(NUM_COMBOS);
    for (int p = 0; p < 2; ++p) values(0, p, range[p], range[1 - p], out, true, ps, 0);
    discount();
  }

  void run(int iterations, uint64_t seed, int threads) {
    if (range[0].empty()) throw std::runtime_error("set ranges first");
    threads = std::max(1, threads);
    if (variant != 0 && threads > 1) throw std::runtime_error("DCFR / CFR+ / PCFR+ run single-threaded (LCFR is the multi-threaded variant)");
    if (threads == 1) {
      Pass ps;
      std::mt19937_64 rng(seed);
      for (int it = 0; it < iterations; ++it) one_iteration(ps, rng);
      return;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      const int n = iterations * (t + 1) / threads - iterations * t / threads;
      pool.emplace_back([this, n, seed, t]() {
        Pass ps;
        std::mt19937_64 rng(seed * 1000003ULL + uint64_t(t) * 7919ULL + 17ULL);
        for (int it = 0; it < n; ++it) one_iteration(ps, rng);
      });
    }
    for (auto& th : pool) th.join();
  }

  // -- strategies ---------------------------------------------------------------------------
  // per infoset key of the node: the average strategy (normalised strategy sums; the current
  // regret-matching strategy where the sums are empty) or the current (final-iterate) strategy
  void strategy_rows(int node_id, bool current, std::vector<float>& out) const {
    const SubgameNode& nd = nodes[node_id];
    const int nk = n_keys(node_id);
    out.assign(size_t(nk) * n_actions, 0.0f);
    for (int k = 0; k < nk; ++k) {
      const size_t info = size_t(info_offset[node_id]) + k;
      double sigma[MAX_ACTIONS], total = 0.0;
      if (!current)
        for (int a = 0; a < n_actions; ++a) total += strat_sum[info * n_actions + a];
      if (!current && total > 0) {
        for (int a = 0; a < n_actions; ++a) out[size_t(k) * n_actions + a] = float(strat_sum[info * n_actions + a] / total);
      } else {
        sigma_of(info, nd.legal, sigma);
        for (int a = 0; a < n_actions; ++a) out[size_t(k) * n_actions + a] = float(sigma[a]);
      }
    }
    if (node_frozen[node_id])
      for (const Frozen& f : frozen)
        if (f.node == node_id)
          for (int a = 0; a < n_actions; ++a) out[size_t(f.hand) * n_actions + a] = a == f.action ? 1.0f : 0.0f;
  }

  py::array_t<float> node_strategy(int node_id, bool current) const {
    if (node_id < 0 || node_id >= int(nodes.size()) || nodes[node_id].kind != 0) throw std::runtime_error("not a decision node");
    std::vector<float> rows;
    strategy_rows(node_id, current, rows);
    py::array_t<float> out({ssize_t(n_keys(node_id)), ssize_t(n_actions)});
    std::memcpy(out.mutable_data(), rows.data(), rows.size() * sizeof(float));
    return out;
  }

  // strategy of every hand at a later-round node given the board cards dealt since the root
  py::array_t<float> strategy_on_board(int node_id, const std::vector<int>& drawn, bool current) const {
    if (node_id < 0 || node_id >= int(nodes.size()) || nodes[node_id].kind != 0) throw std::runtime_error("not a decision node");
    std::vector<float> rows;
    strategy_rows(node_id, current, rows);
    py::array_t<float> out({ssize_t(NUM_COMBOS), ssize_t(n_actions)});
    auto o = out.mutable_unchecked<2>();
    if (node_round[node_id] == root_round) {
      for (int h = 0; h < NUM_COMBOS; ++h)
        for (int a = 0; a < n_actions; ++a) o(h, a) = rows[size_t(h) * n_actions + a];
      return out;
    }
    const int need = BOARD_CARDS_BY_STAGE[node_round[node_id]] - known_board;
    if (int(drawn.size()) < need) throw std::runtime_error("strategy_on_board: not enough board cards for this round");
    const std::vector<uint16_t>& b = bucket_table(node_round[node_id], drawn.data());
    for (int h = 0; h < NUM_COMBOS; ++h)
      for (int a = 0; a < n_actions; ++a) o(h, a) = rows[size_t(b[h]) * n_actions + a];
    return out;
  }

  int child(int node, int action) const {
    if (node < 0 || node >= int(nodes.size()) || action < 0 || action >= n_actions) return -1;
    return nodes[node].child[action];
  }
};

// ------------------------------------------------------------------ tabular blueprint (Pluribus's MCCFR-P)
// The public betting tree of the whole game (every betting sequence of the action abstraction,
// all rounds; a few thousand nodes for our games) x a card abstraction: lossless 169 hand classes
// pre-flop, `buckets` equal-mass buckets of the hand's expected hand strength (equity vs a uniform
// random hand, Monte-Carlo runouts on the flop / turn, exact on the river) on the later rounds.
// Trained with Pluribus's Algorithm 1 (Brown & Sandholm 2019, supplementary material): external-
// sampling MCCFR with unweighted regret updates and periodic linear discounting of regrets and
// strategy counters (Linear MCCFR), negative-regret pruning of the traverser's actions in 95 % of
// the iterations after a warm-up (never on the last betting round or into terminals), a regret
// floor, and the average strategy tracked with sampled action counters (UPDATE-STRATEGY) - here
// on every round, not only the first.  Threads share the tables (benign races, as in Pluribus).
struct Abstraction {
  int buckets = 200;   // per post-flop round
  int samples = 500;   // Monte-Carlo runouts (flop / turn); the river is exact
  int rounds = 4, showdown = 5;
  std::vector<std::vector<float>> edges;  // per round: buckets-1 quantile edges of the EHS (round 0 unused)

  static int preflop_index(int a, int b) {  // 169 classes: 13 pairs, 78 suited, 78 offsuit
    const int ra = a % 13, rb = b % 13, sa = a / 13, sb = b / 13;
    const int hi = std::max(ra, rb), lo = std::min(ra, rb);
    if (hi == lo) return hi;
    const int pair = hi * (hi - 1) / 2 + lo;  // 0..77
    return sa == sb ? 13 + pair : 91 + pair;
  }
  int num_keys(int round) const { return round == 0 ? 169 : buckets; }

  // expected hand strength of (c0, c1) on the first n cards of `board`: P(win) + P(tie)/2 against a
  // uniform random opponent hand over uniform runouts (exact when the board is complete)
  template <class RNG>
  float ehs(int c0, int c1, const int* board, int n, RNG& rng) const {
    bool used[NUM_CARDS] = {};
    used[c0] = used[c1] = true;
    for (int i = 0; i < n; ++i) used[board[i]] = true;
    int deck[NUM_CARDS], nd = 0;
    for (int c = 0; c < NUM_CARDS; ++c)
      if (!used[c]) deck[nd++] = c;
    const int missing = showdown - n;
    int full[7] = {c0, c1, 0, 0, 0, 0, 0}, opp[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n; ++i) full[2 + i] = opp[2 + i] = board[i];
    double win = 0.0, cnt = 0.0;
    if (missing == 0) {
      const int mine = eval_best(full, 2 + showdown);
      for (int i = 0; i < nd; ++i)
        for (int j = i + 1; j < nd; ++j) {
          opp[0] = deck[i]; opp[1] = deck[j];
          const int s = eval_best(opp, 2 + showdown);
          win += mine < s ? 1.0 : mine == s ? 0.5 : 0.0;
          cnt += 1.0;
        }
      return float(win / cnt);
    }
    for (int s_i = 0; s_i < samples; ++s_i) {  // sample runout + opponent hand without replacement
      for (int i = 0; i < missing + 2; ++i) {
        std::uniform_int_distribution<int> d(i, nd - 1);
        std::swap(deck[i], deck[d(rng)]);
      }
      for (int i = 0; i < missing; ++i) full[2 + n + i] = opp[2 + n + i] = deck[i];
      opp[0] = deck[missing]; opp[1] = deck[missing + 1];
      const int mine = eval_best(full, 2 + showdown), s = eval_best(opp, 2 + showdown);
      win += mine < s ? 1.0 : mine == s ? 0.5 : 0.0;
      cnt += 1.0;
    }
    return float(win / cnt);
  }

  // EHS of every combo at once (shared runouts): exact on a complete board, otherwise the mean over
  // `runouts` sampled completions of the BoardTable equities (hands blocked by a runout skip it).
  // Used for hand-substituted queries (1326 hands of one public state); ~100x cheaper than per hand.
  template <class RNG>
  void ehs_all(const int* board, int n, int runouts, RNG& rng, std::vector<float>& out) const {
    out.assign(NUM_COMBOS, -1.0f);
    const int missing = showdown - n;
    if (missing == 0) {
      BoardTable t;
      t.build(board, showdown);
      out = t.equity;
      return;
    }
    bool used[NUM_CARDS] = {};
    for (int i = 0; i < n; ++i) used[board[i]] = true;
    int deck[NUM_CARDS], nd = 0;
    for (int c = 0; c < NUM_CARDS; ++c)
      if (!used[c]) deck[nd++] = c;
    std::vector<double> acc(NUM_COMBOS, 0.0), cnt(NUM_COMBOS, 0.0);
    int full[5];
    for (int i = 0; i < n; ++i) full[i] = board[i];
    BoardTable t;
    for (int r = 0; r < runouts; ++r) {
      for (int i = 0; i < missing; ++i) {
        std::uniform_int_distribution<int> d(i, nd - 1);
        std::swap(deck[i], deck[d(rng)]);
        full[n + i] = deck[i];
      }
      t.build(full, showdown);
      for (int h : t.order) { acc[h] += t.equity[h]; cnt[h] += 1.0; }
    }
    for (int h = 0; h < NUM_COMBOS; ++h)
      if (cnt[h] > 0) out[h] = float(acc[h] / cnt[h]);
  }

  int bucket_of(int round, float e) const {
    const std::vector<float>& ed = edges[round];
    return int(std::upper_bound(ed.begin(), ed.end(), e) - ed.begin());
  }
  template <class RNG>
  int bucket(int round, int c0, int c1, const int* board, RNG& rng) const {
    if (round == 0) return preflop_index(c0, c1);
    return bucket_of(round, ehs(c0, c1, board, BOARD_CARDS_BY_STAGE[round], rng));
  }

  // equal-mass bucket edges from random situations of every post-flop round
  void fit_edges(int situations, uint64_t seed, int threads) {
    edges.assign(rounds, {});
    for (int r = 1; r < rounds; ++r) {
      std::vector<float> vals(situations);
      const int nb = BOARD_CARDS_BY_STAGE[r];
      auto work = [&](int lo, int hi, uint64_t s) {
        std::mt19937_64 rng(s);
        for (int k = lo; k < hi; ++k) {
          int deck[NUM_CARDS];
          for (int i = 0; i < NUM_CARDS; ++i) deck[i] = i;
          for (int i = 0; i < 2 + nb; ++i) {
            std::uniform_int_distribution<int> d(i, NUM_CARDS - 1);
            std::swap(deck[i], deck[d(rng)]);
          }
          vals[k] = ehs(deck[0], deck[1], deck + 2, nb, rng);
        }
      };
      const int nt = std::max(1, threads);
      std::vector<std::thread> pool;
      for (int t = 0; t < nt; ++t)
        pool.emplace_back(work, situations * t / nt, situations * (t + 1) / nt, seed * 7919ULL + uint64_t(r) * 104729ULL + uint64_t(t));
      for (auto& th : pool) th.join();
      std::sort(vals.begin(), vals.end());
      for (int b = 1; b < buckets; ++b) edges[r].push_back(vals[size_t(situations) * b / buckets]);
    }
  }
};

struct TabularBlueprint {
  EngineConfig cfg;
  PublicTree tree;
  Abstraction abs;
  int n_actions = 4;
  std::vector<int> info_offset;
  size_t n_infosets = 0;
  std::vector<float> regret, phi;  // per infoset x action; phi = average-strategy counters
  std::atomic<long long> iteration{0};
  // Pluribus parameters (iterations instead of minutes)
  double prune_threshold = -3e6, regret_floor = -3.1e6;
  long long prune_after = 0, lcfr_iterations = 0, discount_interval = 0, strategy_interval = 10000;
  double prune_prob = 0.95;
  std::mutex discount_mutex;

  void build(const EngineConfig& c, int buckets, int samples) {
    cfg = c;
    Engine root(cfg);
    int deck[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    root.reset(deck);
    n_actions = root.num_actions();
    tree.build(root);
    abs.buckets = buckets;
    abs.samples = samples;
    abs.rounds = cfg.num_rounds;
    abs.showdown = cfg.showdown_cards();
    info_offset.assign(tree.nodes.size(), 0);
    n_infosets = 0;
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
      info_offset[i] = int(n_infosets);
      if (tree.nodes[i].kind == 0) n_infosets += size_t(abs.num_keys(tree.round[i]));
    }
    regret.assign(n_infosets * n_actions, 0.0f);
    phi.assign(n_infosets * n_actions, 0.0f);
    iteration = 0;
  }

  void sigma_of(size_t info, const bool* legal, float* sigma) const {
    const float* R = regret.data() + info * n_actions;
    float total = 0.0f;
    int cnt = 0;
    for (int a = 0; a < n_actions; ++a) {
      const float r = (legal[a] && R[a] > 0) ? R[a] : 0.0f;
      sigma[a] = r;
      total += r;
      cnt += legal[a];
    }
    for (int a = 0; a < n_actions; ++a) sigma[a] = total > 0 ? sigma[a] / total : (legal[a] ? 1.0f / cnt : 0.0f);
  }

  // -- one traversal ----------------------------------------------------------------------
  struct Deal {
    int hand[2][2], board[5];
    int win[2];             // +1 / -1 / 0 at showdown for each seat
    int bucket[2][4];       // per seat and round, -1 = not yet computed
    bool prune;
    std::mt19937_64* rng;
  };

  int key_of(Deal& d, int seat, int round) const {
    int& b = d.bucket[seat][round];
    if (b < 0) b = abs.bucket(round, d.hand[seat][0], d.hand[seat][1], d.board, *d.rng);
    return b;
  }

  float traverse(int node, int p, Deal& d) {
    const SubgameNode& nd = tree.nodes[node];
    if (nd.kind == 1) return nd.folder == p ? -float(nd.stake) : float(nd.stake);
    if (nd.kind == 2) return float(nd.stake) * float(d.win[p]);
    const int round = tree.round[node];
    const size_t info = size_t(info_offset[node]) + key_of(d, nd.player, round);
    float sigma[MAX_ACTIONS];
    sigma_of(info, nd.legal, sigma);
    if (nd.player == p) {
      float v[MAX_ACTIONS], value = 0.0f;
      bool explored[MAX_ACTIONS] = {};
      float* R = regret.data() + info * n_actions;
      const bool last = round == cfg.num_rounds - 1;
      for (int a = 0; a < n_actions; ++a) {
        if (!nd.legal[a]) continue;
        const bool terminal_child = tree.nodes[nd.child[a]].kind != 0;
        if (d.prune && !last && !terminal_child && R[a] <= prune_threshold) continue;
        v[a] = traverse(nd.child[a], p, d);
        explored[a] = true;
        value += sigma[a] * v[a];
      }
      for (int a = 0; a < n_actions; ++a)
        if (explored[a]) R[a] = std::max(float(regret_floor), R[a] + v[a] - value);
      return value;
    }
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float x = u(*d.rng), acc = 0.0f;
    int chosen = -1;
    for (int a = 0; a < n_actions; ++a) {
      if (!nd.legal[a]) continue;
      acc += sigma[a];
      chosen = a;
      if (x < acc) break;
    }
    return traverse(nd.child[chosen], p, d);
  }

  void update_strategy(int node, int p, Deal& d) {  // Pluribus UPDATE-STRATEGY (all rounds here)
    const SubgameNode& nd = tree.nodes[node];
    if (nd.kind != 0) return;
    if (nd.player == p) {
      const size_t info = size_t(info_offset[node]) + key_of(d, p, tree.round[node]);
      float sigma[MAX_ACTIONS];
      sigma_of(info, nd.legal, sigma);
      std::uniform_real_distribution<float> u(0.0f, 1.0f);
      float x = u(*d.rng), acc = 0.0f;
      int chosen = -1;
      for (int a = 0; a < n_actions; ++a) {
        if (!nd.legal[a]) continue;
        acc += sigma[a];
        chosen = a;
        if (x < acc) break;
      }
      phi[info * n_actions + chosen] += 1.0f;
      update_strategy(nd.child[chosen], p, d);
      return;
    }
    for (int a = 0; a < n_actions; ++a)
      if (nd.legal[a]) update_strategy(nd.child[a], p, d);
  }

  void deal(Deal& d, std::mt19937_64& rng) {
    int deck[NUM_CARDS];
    for (int i = 0; i < NUM_CARDS; ++i) deck[i] = i;
    for (int i = 0; i < 9; ++i) {
      std::uniform_int_distribution<int> u(i, NUM_CARDS - 1);
      std::swap(deck[i], deck[u(rng)]);
    }
    d.hand[0][0] = deck[0]; d.hand[0][1] = deck[1]; d.hand[1][0] = deck[2]; d.hand[1][1] = deck[3];
    for (int i = 0; i < 5; ++i) d.board[i] = deck[4 + i];
    const int nb = abs.showdown;
    int c0[7] = {d.hand[0][0], d.hand[0][1], d.board[0], d.board[1], d.board[2], d.board[3], d.board[4]};
    int c1[7] = {d.hand[1][0], d.hand[1][1], d.board[0], d.board[1], d.board[2], d.board[3], d.board[4]};
    const int s0 = eval_best(c0, 2 + nb), s1 = eval_best(c1, 2 + nb);
    d.win[0] = s0 == s1 ? 0 : (s0 < s1 ? 1 : -1);
    d.win[1] = -d.win[0];
    for (int s = 0; s < 2; ++s)
      for (int r = 0; r < 4; ++r) d.bucket[s][r] = -1;
    d.rng = &rng;
  }

  void discount(double f) {
    for (size_t i = 0; i < regret.size(); ++i) regret[i] = float(regret[i] * f);
    for (size_t i = 0; i < phi.size(); ++i) phi[i] = float(phi[i] * f);
  }

  void run(long long iterations, uint64_t seed, int threads) {
    threads = std::max(1, threads);
    auto worker = [this, iterations, seed, threads](int t) {
      const long long n = iterations * (t + 1) / threads - iterations * t / threads;
      std::mt19937_64 rng(seed * 1000003ULL + uint64_t(t) * 7919ULL + 3ULL);
      std::uniform_real_distribution<double> u01(0.0, 1.0);
      Deal d;
      for (long long it = 0; it < n; ++it) {
        const long long tt = ++iteration;
        for (int p = 0; p < 2; ++p) {
          deal(d, rng);
          d.prune = tt > prune_after && prune_after > 0 && u01(rng) < prune_prob;
          traverse(0, p, d);
          if (strategy_interval > 0 && tt % strategy_interval == 0) {
            deal(d, rng);
            update_strategy(0, p, d);
          }
        }
        if (discount_interval > 0 && tt < lcfr_iterations && tt % discount_interval == 0) {
          const double k = double(tt / discount_interval);
          std::lock_guard<std::mutex> lock(discount_mutex);
          discount(k / (k + 1.0));
        }
      }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
  }

  // -- queries --------------------------------------------------------------------------------
  int child(int node, int a) const {
    if (node < 0 || node >= int(tree.nodes.size()) || a < 0 || a >= n_actions) return -1;
    return tree.nodes[node].child[a];
  }

  // strategy at a node for one hand: normalised average counters (phi), regret matching where empty
  void strategy_at(int node, int key, bool current, float* out) const {
    const SubgameNode& nd = tree.nodes[node];
    const size_t info = size_t(info_offset[node]) + key;
    if (!current) {
      float total = 0.0f;
      for (int a = 0; a < n_actions; ++a) total += nd.legal[a] ? phi[info * n_actions + a] : 0.0f;
      if (total > 0) {
        for (int a = 0; a < n_actions; ++a) out[a] = nd.legal[a] ? phi[info * n_actions + a] / total : 0.0f;
        return;
      }
    }
    sigma_of(info, nd.legal, out);
  }
};

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
    if (cards.size() < 5 || cards.size() > 7) throw std::runtime_error("need 5..7 cards");
    return eval_best(cards.data(), int(cards.size()));
  });

  py::class_<EngineConfig>(m, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("stack_size", &EngineConfig::stack_size)
      .def_readwrite("small_blind", &EngineConfig::small_blind)
      .def_readwrite("big_blind", &EngineConfig::big_blind)
      .def_readwrite("raise_cap", &EngineConfig::raise_cap)
      .def_readwrite("bet_sizes", &EngineConfig::bet_sizes)
      .def_readwrite("mask_redundant", &EngineConfig::mask_redundant)
      .def_readwrite("limit", &EngineConfig::limit)
      .def_readwrite("raise_caps", &EngineConfig::raise_caps)
      .def_readwrite("num_rounds", &EngineConfig::num_rounds)
      .def_readwrite("has_all_in", &EngineConfig::has_all_in)
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

  py::class_<SubgameSolver>(m, "SubgameSolver")
      .def(py::init<>())
      .def("build", [](SubgameSolver& sv, const Engine& root) {
        if (root.done) throw std::runtime_error("the root state is terminal");
        sv.build(root);
      })
      .def("set_ranges", [](SubgameSolver& sv, py::array_t<float, py::array::c_style | py::array::forcecast> r0,
                            py::array_t<float, py::array::c_style | py::array::forcecast> r1) {
        if (r0.size() != NUM_COMBOS || r1.size() != NUM_COMBOS) throw std::runtime_error("ranges must have 1326 entries");
        sv.set_ranges(r0.data(), r1.data());
      })
      .def("set_continuations", [](SubgameSolver& sv, std::vector<std::shared_ptr<Model>> nets0, std::vector<std::shared_ptr<Model>> nets1,
                                   std::vector<bool> rm, std::vector<double> weights) {
        if (nets0.size() != nets1.size() || nets0.size() != rm.size() || nets0.size() != weights.size() || nets0.empty())
          throw std::runtime_error("continuations: need equally many seat-0 nets, seat-1 nets, rm flags and weights");
        sv.conts.clear();
        sv.cont_cum.clear();
        double acc = 0.0;
        for (size_t i = 0; i < nets0.size(); ++i) {
          sv.conts.push_back({{nets0[i].get(), nets1[i].get()}, rm[i], weights[i]});
          acc += weights[i];
          sv.cont_cum.push_back(acc);
        }
        sv.keep_alive = {nets0, nets1};
      })
      .def("run", [](SubgameSolver& sv, int iterations, uint64_t seed, int hero_seat, int hero_hand, double focus) {
        if (sv.conts.empty() && std::any_of(sv.nodes.begin(), sv.nodes.end(), [](const SubgameNode& n) { return n.kind == 3; }))
          throw std::runtime_error("the subgame has street-end leaves: set continuation strategies first");
        if (sv.range[0].empty()) throw std::runtime_error("set ranges first");
        if (hero_hand >= 0 && sv.range[hero_seat][hero_hand] <= 0) throw std::runtime_error("the hero's hand has zero weight in its range");
        py::gil_scoped_release release;
        sv.run(iterations, seed, hero_seat, hero_hand, focus);
      }, py::arg("iterations"), py::arg("seed") = 0, py::arg("hero_seat") = 0, py::arg("hero_hand") = -1, py::arg("focus") = 0.0)
      .def("set_variant", [](SubgameSolver& sv, const std::string& name, double alpha, double beta, double gamma) {
        if (name == "lcfr") sv.variant = 0;
        else if (name == "dcfr") sv.variant = 1;
        else if (name == "cfr+") sv.variant = 2;
        else if (name == "pcfr+") sv.variant = 3;
        else throw std::runtime_error("variant must be lcfr | dcfr | cfr+ | pcfr+");
        sv.alpha = alpha; sv.beta = beta; sv.gamma = gamma;
      }, py::arg("name"), py::arg("alpha") = 1.5, py::arg("beta") = 0.0, py::arg("gamma") = 2.0)
      .def("root_strategy", [](SubgameSolver& sv) { return sv.node_strategy(0); })
      .def("node_strategy", &SubgameSolver::node_strategy)
      .def("warm_start", &SubgameSolver::warm_start, py::arg("previous"), py::arg("old_root"), py::arg("weight") = 2000)
      .def("child", [](const SubgameSolver& sv, int node, int action) {
        if (node < 0 || node >= int(sv.nodes.size()) || action < 0 || action >= sv.n_actions) throw std::runtime_error("bad node / action");
        const SubgameNode& nd = sv.nodes[node];
        if (nd.kind != 0) return -1;
        return nd.legal[action] ? nd.child[action] : nd.child[nd.twin[action]];
      })
      .def("tree", [](const SubgameSolver& sv) {
        py::list out;
        for (const SubgameNode& nd : sv.nodes) {
          py::dict d;
          d["kind"] = nd.kind;
          d["player"] = nd.player;
          d["folder"] = nd.folder;
          d["stake"] = nd.stake;
          d["legal"] = std::vector<bool>(nd.legal, nd.legal + sv.n_actions);
          d["twin"] = std::vector<int>(nd.twin, nd.twin + sv.n_actions);
          d["child"] = std::vector<int>(nd.child, nd.child + sv.n_actions);
          d["stage"] = nd.state.stage;
          out.append(d);
        }
        return out;
      })
      .def_property_readonly("num_nodes", [](const SubgameSolver& sv) { return int(sv.nodes.size()); })
      .def_property_readonly("num_leaves", [](const SubgameSolver& sv) {
        return int(std::count_if(sv.nodes.begin(), sv.nodes.end(), [](const SubgameNode& n) { return n.kind == 3; }));
      })
      .def_property_readonly("iterations", [](const SubgameSolver& sv) { return sv.iteration; });

  py::class_<BoardTable>(m, "BoardTable", "strengths / equities of all 1326 combos on a complete board and the vector-form showdown payoffs")
      .def(py::init([](const std::vector<int>& board, int nb) {
        if (nb < 3 || nb > 5 || int(board.size()) < nb) throw std::runtime_error("BoardTable needs the 3..5 cards of the complete board");
        auto t = std::make_unique<BoardTable>();
        int b[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < nb; ++i) b[i] = board[i];
        t->build(b, nb);
        return t;
      }), py::arg("board"), py::arg("num_cards") = 5)
      .def("showdown_values", [](const BoardTable& t, py::array_t<double, py::array::c_style | py::array::forcecast> reach) {
        if (reach.size() != NUM_COMBOS) throw std::runtime_error("reach must have 1326 entries");
        std::vector<double> r(reach.data(), reach.data() + NUM_COMBOS), out(NUM_COMBOS);
        t.showdown_values(r, out);
        return py::array_t<double>(NUM_COMBOS, out.data());
      })
      .def_static("opponent_mass", [](py::array_t<double, py::array::c_style | py::array::forcecast> reach) {
        if (reach.size() != NUM_COMBOS) throw std::runtime_error("reach must have 1326 entries");
        std::vector<double> r(reach.data(), reach.data() + NUM_COMBOS), out(NUM_COMBOS);
        BoardTable::opponent_mass(r, out);
        return py::array_t<double>(NUM_COMBOS, out.data());
      })
      .def_property_readonly("strength", [](const BoardTable& t) { return py::array_t<int>(NUM_COMBOS, t.strength.data()); })
      .def_property_readonly("equity", [](const BoardTable& t) { return py::array_t<float>(NUM_COMBOS, t.equity.data()); });

  py::class_<VectorSolver>(m, "VectorSolver")
      .def(py::init<>())
      .def("build", [](VectorSolver& sv, const Engine& root, int buckets) {
        if (buckets < 1 || buckets > 65535) throw std::runtime_error("buckets must be in 1..65535");
        sv.buckets = buckets;
        py::gil_scoped_release release;
        sv.build(root);
      }, py::arg("root"), py::arg("buckets") = 500)
      .def("set_ranges", [](VectorSolver& sv, py::array_t<float, py::array::c_style | py::array::forcecast> r0,
                            py::array_t<float, py::array::c_style | py::array::forcecast> r1) {
        if (r0.size() != NUM_COMBOS || r1.size() != NUM_COMBOS) throw std::runtime_error("ranges must have 1326 entries");
        sv.set_ranges(r0.data(), r1.data());
      })
      .def("set_variant", [](VectorSolver& sv, const std::string& name, double alpha, double beta, double gamma) {
        if (name == "lcfr") sv.variant = 0;
        else if (name == "dcfr") sv.variant = 1;
        else if (name == "cfr+") sv.variant = 2;
        else if (name == "pcfr+") sv.variant = 3;
        else throw std::runtime_error("variant must be lcfr | dcfr | cfr+ | pcfr+");
        sv.alpha = alpha; sv.beta = beta; sv.gamma = gamma;
      }, py::arg("name"), py::arg("alpha") = 1.5, py::arg("beta") = 0.0, py::arg("gamma") = 2.0)
      .def("freeze", &VectorSolver::freeze, py::arg("node"), py::arg("hand"), py::arg("action"))
      .def("run", [](VectorSolver& sv, int iterations, uint64_t seed, int threads) {
        py::gil_scoped_release release;
        sv.run(iterations, seed, threads);
      }, py::arg("iterations"), py::arg("seed") = 0, py::arg("threads") = 1)
      .def("root_strategy", [](const VectorSolver& sv, bool current) { return sv.node_strategy(0, current); }, py::arg("current") = false)
      .def("node_strategy", &VectorSolver::node_strategy, py::arg("node"), py::arg("current") = false)
      .def("strategy_on_board", &VectorSolver::strategy_on_board, py::arg("node"), py::arg("drawn"), py::arg("current") = false)
      .def("buckets_of", [](const VectorSolver& sv, int round, const std::vector<int>& drawn) {
        if (round <= sv.root_round || round > sv.last_round) throw std::runtime_error("buckets exist for the rounds after the root round");
        if (int(drawn.size()) < BOARD_CARDS_BY_STAGE[round] - sv.known_board) throw std::runtime_error("not enough board cards for this round");
        const std::vector<uint16_t>& b = sv.bucket_table(round, drawn.data());
        return std::vector<int>(b.begin(), b.end());
      }, py::arg("round"), py::arg("drawn"))
      .def("child", &VectorSolver::child)
      .def("node_player", [](const VectorSolver& sv, int node) {
        if (node < 0 || node >= int(sv.nodes.size())) throw std::runtime_error("bad node");
        return sv.nodes[node].kind == 0 ? sv.nodes[node].player : -1;
      })
      .def("node_round", [](const VectorSolver& sv, int node) {
        if (node < 0 || node >= int(sv.nodes.size())) throw std::runtime_error("bad node");
        return sv.node_round[node];
      })
      .def("tree", [](const VectorSolver& sv) {
        py::list out;
        for (size_t i = 0; i < sv.nodes.size(); ++i) {
          const SubgameNode& nd = sv.nodes[i];
          py::dict d;
          d["kind"] = nd.kind; d["player"] = nd.player; d["folder"] = nd.folder; d["stake"] = nd.stake;
          d["legal"] = std::vector<bool>(nd.legal, nd.legal + sv.n_actions);
          d["twin"] = std::vector<int>(nd.twin, nd.twin + sv.n_actions);
          d["child"] = std::vector<int>(nd.child, nd.child + sv.n_actions);
          d["stage"] = nd.state.stage;
          d["round"] = sv.node_round[i];
          out.append(d);
        }
        return out;
      })
      .def_property_readonly("num_nodes", [](const VectorSolver& sv) { return int(sv.nodes.size()); })
      .def_property_readonly("num_infosets", [](const VectorSolver& sv) { return int(sv.n_infosets); })
      .def_property_readonly("root_round", [](const VectorSolver& sv) { return sv.root_round; })
      .def_property_readonly("iterations", [](const VectorSolver& sv) { return sv.iteration.load(); });

  py::class_<TabularBlueprint>(m, "TabularBlueprint")
      .def(py::init<>())
      .def("build", [](TabularBlueprint& b, const EngineConfig& cfg, int buckets, int samples) {
        if (buckets < 1 || buckets > 65535 || samples < 1) throw std::runtime_error("bad buckets / samples");
        b.build(cfg, buckets, samples);
      }, py::arg("cfg"), py::arg("buckets") = 200, py::arg("samples") = 500)
      .def("fit_abstraction", [](TabularBlueprint& b, int situations, uint64_t seed, int threads) {
        py::gil_scoped_release release;
        b.abs.fit_edges(situations, seed, threads);
      }, py::arg("situations") = 200000, py::arg("seed") = 0, py::arg("threads") = 16)
      .def("set_params", [](TabularBlueprint& b, double prune_threshold, double regret_floor, long long prune_after,
                            long long lcfr_iterations, long long discount_interval, long long strategy_interval, double prune_prob) {
        b.prune_threshold = prune_threshold; b.regret_floor = regret_floor; b.prune_after = prune_after;
        b.lcfr_iterations = lcfr_iterations; b.discount_interval = discount_interval; b.strategy_interval = strategy_interval;
        b.prune_prob = prune_prob;
      }, py::arg("prune_threshold"), py::arg("regret_floor"), py::arg("prune_after"), py::arg("lcfr_iterations"),
         py::arg("discount_interval"), py::arg("strategy_interval") = 10000, py::arg("prune_prob") = 0.95)
      .def("run", [](TabularBlueprint& b, long long iterations, uint64_t seed, int threads) {
        if (b.abs.edges.empty() && b.cfg.num_rounds > 1) throw std::runtime_error("fit_abstraction first");
        py::gil_scoped_release release;
        b.run(iterations, seed, threads);
      }, py::arg("iterations"), py::arg("seed") = 0, py::arg("threads") = 16)
      .def("child", &TabularBlueprint::child)
      .def("node_player", [](const TabularBlueprint& b, int node) { return b.tree.nodes.at(node).kind == 0 ? b.tree.nodes[node].player : -1; })
      .def("node_round", [](const TabularBlueprint& b, int node) { return b.tree.round.at(node); })
      .def("node_legal", [](const TabularBlueprint& b, int node) {
        const SubgameNode& nd = b.tree.nodes.at(node);
        return std::vector<bool>(nd.legal, nd.legal + b.n_actions);
      })
      .def("bucket", [](const TabularBlueprint& b, int round, int c0, int c1, const std::vector<int>& board, uint64_t seed) {
        std::mt19937_64 rng(seed);
        int bd[5] = {0, 0, 0, 0, 0};
        for (size_t i = 0; i < board.size() && i < 5; ++i) bd[i] = board[i];
        return b.abs.bucket(round, c0, c1, bd, rng);
      }, py::arg("round"), py::arg("c0"), py::arg("c1"), py::arg("board"), py::arg("seed") = 0)
      .def("ehs", [](const TabularBlueprint& b, int c0, int c1, const std::vector<int>& board, uint64_t seed) {
        std::mt19937_64 rng(seed);
        int bd[5] = {0, 0, 0, 0, 0};
        for (size_t i = 0; i < board.size() && i < 5; ++i) bd[i] = board[i];
        return b.abs.ehs(c0, c1, bd, int(std::min<size_t>(board.size(), 5)), rng);
      }, py::arg("c0"), py::arg("c1"), py::arg("board"), py::arg("seed") = 0)
      .def("strategy", [](const TabularBlueprint& b, int node, int key, bool current) {
        if (node < 0 || node >= int(b.tree.nodes.size()) || b.tree.nodes[node].kind != 0) throw std::runtime_error("not a decision node");
        if (key < 0 || key >= b.abs.num_keys(b.tree.round[node])) throw std::runtime_error("bad key");
        py::array_t<float> out(b.n_actions);
        b.strategy_at(node, key, current, out.mutable_data());
        return out;
      }, py::arg("node"), py::arg("key"), py::arg("current") = false)
      .def("strategy_for_hands", [](const TabularBlueprint& b, int node, py::array_t<int, py::array::c_style | py::array::forcecast> hands,
                                    const std::vector<int>& board, uint64_t seed, bool current) {
        if (node < 0 || node >= int(b.tree.nodes.size()) || b.tree.nodes[node].kind != 0) throw std::runtime_error("not a decision node");
        if (hands.ndim() != 2 || hands.shape(1) != 2) throw std::runtime_error("hands must be (N, 2)");
        const int n = int(hands.shape(0)), round = b.tree.round[node];
        int bd[5] = {0, 0, 0, 0, 0};
        for (size_t i = 0; i < board.size() && i < 5; ++i) bd[i] = board[i];
        py::array_t<float> out({ssize_t(n), ssize_t(b.n_actions)});
        auto h = hands.unchecked<2>();
        auto o = out.mutable_unchecked<2>();
        py::gil_scoped_release release;
        std::mt19937_64 rng(seed);
        std::vector<float> row(b.n_actions), all;
        const int nb = BOARD_CARDS_BY_STAGE[round];
        const bool vector_path = round > 0 && n >= 64;  // many hands of one state: shared runouts for all combos
        if (vector_path) b.abs.ehs_all(bd, nb, std::max(20, b.abs.samples / 5), rng, all);
        for (int i = 0; i < n; ++i) {
          bool blocked = h(i, 0) == h(i, 1);
          for (int k = 0; k < nb; ++k) blocked |= (bd[k] == h(i, 0) || bd[k] == h(i, 1));
          if (blocked) {  // impossible hand (hand-substituted queries): zeros, the caller masks / normalises
            for (int a = 0; a < b.n_actions; ++a) o(i, a) = 0.0f;
            continue;
          }
          int key;
          if (vector_path) {
            const int lo = std::min(h(i, 0), h(i, 1)), hi = std::max(h(i, 0), h(i, 1));
            key = b.abs.bucket_of(round, all[combo_index(lo, hi)]);
          } else {
            key = b.abs.bucket(round, h(i, 0), h(i, 1), bd, rng);
          }
          b.strategy_at(node, key, current, row.data());
          for (int a = 0; a < b.n_actions; ++a) o(i, a) = row[a];
        }
        return out;
      }, py::arg("node"), py::arg("hands"), py::arg("board"), py::arg("seed") = 0, py::arg("current") = false)
      .def_property("regret", [](const TabularBlueprint& b) { return py::array_t<float>(b.regret.size(), b.regret.data()); },
                    [](TabularBlueprint& b, py::array_t<float, py::array::c_style | py::array::forcecast> a) {
                      if (size_t(a.size()) != b.regret.size()) throw std::runtime_error("regret size mismatch");
                      std::memcpy(b.regret.data(), a.data(), b.regret.size() * sizeof(float));
                    })
      .def_property("phi", [](const TabularBlueprint& b) { return py::array_t<float>(b.phi.size(), b.phi.data()); },
                    [](TabularBlueprint& b, py::array_t<float, py::array::c_style | py::array::forcecast> a) {
                      if (size_t(a.size()) != b.phi.size()) throw std::runtime_error("phi size mismatch");
                      std::memcpy(b.phi.data(), a.data(), b.phi.size() * sizeof(float));
                    })
      .def_property("edges", [](const TabularBlueprint& b) { return b.abs.edges; },
                    [](TabularBlueprint& b, const std::vector<std::vector<float>>& e) {
                      if (int(e.size()) != b.abs.rounds) throw std::runtime_error("edges: one list per round");
                      b.abs.edges = e;
                    })
      .def_property("iterations", [](const TabularBlueprint& b) { return b.iteration.load(); },
                    [](TabularBlueprint& b, long long v) { b.iteration = v; })
      .def_property_readonly("num_nodes", [](const TabularBlueprint& b) { return int(b.tree.nodes.size()); })
      .def_property_readonly("num_infosets", [](const TabularBlueprint& b) { return int(b.n_infosets); })
      .def_property_readonly("num_actions", [](const TabularBlueprint& b) { return b.n_actions; })
      .def_property_readonly("buckets", [](const TabularBlueprint& b) { return b.abs.buckets; })
      .def_property_readonly("samples", [](const TabularBlueprint& b) { return b.abs.samples; });

  m.def("run_dream", &run_dream, py::arg("net0"), py::arg("net1"), py::arg("baseline"), py::arg("traverser"), py::arg("n_traversals"),
        py::arg("t"), py::arg("epsilon"), py::arg("seed"), py::arg("cfg") = EngineConfig(), py::arg("decks") = py::none(),
        "DREAM outcome-sampling traversals: (adv obs, adv weight, adv target, history obs, action, Q target, nodes)");
  m.def("run_escher_values", &run_escher_values, py::arg("net0"), py::arg("net1"), py::arg("n_trajectories"), py::arg("seed"),
        py::arg("cfg") = EngineConfig(), "ESCHER value trajectories: (history obs, action, u_0 target, nodes)");
  m.def("run_escher_regrets", &run_escher_regrets, py::arg("net0"), py::arg("net1"), py::arg("vnet"), py::arg("traverser"),
        py::arg("n_trajectories"), py::arg("t"), py::arg("seed"), py::arg("cfg") = EngineConfig(),
        "ESCHER regret trajectories: (adv obs, t, target, strat obs, t, sigma, history obs, -1, value, nodes)");
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
