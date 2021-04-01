// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dpf/distributed_point_function.h"

#include <glog/logging.h>
#include <openssl/rand.h>

#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "dpf/internal/array_conversions.h"
#include "dpf/status_macros.h"

namespace private_statistics {
namespace dpf {

namespace {

// PRG keys used to expand seeds using AES. The first two are used to compute
// correction words of seeds, while the last is used to compute correction
// words of the incremental DPF values. Values were computed by taking the
// first half of the SHA256 sum of the constant name, e.g., `echo
// "DistributedPointFunction::kPrgKeyLeft" | sha256sum`
constexpr absl::uint128 kPrgKeyLeft =
    absl::MakeUint128(0x5be037ccf6a03de5ULL, 0x935f08d0a5b6a2fdULL);
constexpr absl::uint128 kPrgKeyRight =
    absl::MakeUint128(0xef94b6aedebb026cULL, 0xe2ea1fe0f66f4d0bULL);
constexpr absl::uint128 kPrgKeyValue =
    absl::MakeUint128(0x05a5d1588c5423e3ULL, 0x46a31101b21d1c98ULL);

// Extracts the lowest bit of `x` and sets it to 0 in `x`.
bool ExtractAndClearLowestBit(absl::uint128& x) {
  bool bit = ((x & absl::uint128{1}) != 0);
  x &= ~absl::uint128{1};
  return bit;
}

// Computes the value correction word given two seeds `seed_a`, `seed_b` for
// parties a and b, such that the element at `block_index` is equal to `beta`.
// If `invert` is true, the result is multiplied element-wise by -1. Templated
// to use the correct integer type without needing modular reduction.
template <typename T>
absl::uint128 ComputeValueCorrectionFor(absl::Span<const absl::uint128> seeds,
                                        int block_index, T beta, bool invert) {
  constexpr int elements_per_block = dpf_internal::ElementsPerBlock<T>();

  // Split up seeds into individual integers.
  std::array<T, elements_per_block> ints_a = dpf_internal::Uint128ToArray<T>(
                                        seeds[0]),
                                    ints_b = dpf_internal::Uint128ToArray<T>(
                                        seeds[1]);

  // Add beta to the right position.
  ints_b[block_index] += beta;

  // Add up shares, invert if needed.
  for (int i = 0; i < elements_per_block; i++) {
    ints_b[i] = ints_b[i] - ints_a[i];
    if (invert) {
      ints_b[i] = -ints_b[i];
    }
  }

  // Re-assemble block.
  return dpf_internal::ArrayToUint128(ints_b);
}

}  // namespace

DistributedPointFunction::DistributedPointFunction(
    std::vector<DpfParameters> parameters, int tree_levels_needed,
    absl::flat_hash_map<int, int> tree_to_hierarchy,
    std::vector<int> hierarchy_to_tree,
    dpf_internal::PseudorandomGenerator prg_left,
    dpf_internal::PseudorandomGenerator prg_right,
    dpf_internal::PseudorandomGenerator prg_value)
    : parameters_(std::move(parameters)),
      tree_levels_needed_(tree_levels_needed),
      tree_to_hierarchy_(std::move(tree_to_hierarchy)),
      hierarchy_to_tree_(std::move(hierarchy_to_tree)),
      prg_left_(std::move(prg_left)),
      prg_right_(std::move(prg_right)),
      prg_value_(std::move(prg_value)) {}

absl::StatusOr<absl::uint128> DistributedPointFunction::ComputeValueCorrection(
    int hierarchy_level, absl::Span<const absl::uint128> seeds,
    absl::uint128 alpha, absl::uint128 beta, bool invert) const {
  // Compute third PRG output component of current seed.
  std::array<absl::uint128, 2> value_correction_shares;
  DPF_RETURN_IF_ERROR(
      prg_value_.Evaluate(seeds, absl::MakeSpan(value_correction_shares)));

  // Compute index in block for alpha at the current hierarchy level.
  int index_in_block = DomainToBlockIndex(alpha, hierarchy_level);

  // Choose implementation depending on element_bitsize.
  int element_bitsize = parameters_[hierarchy_level].element_bitsize();
  absl::uint128 value_correction;
  switch (element_bitsize) {
    case 8:
      value_correction =
          ComputeValueCorrectionFor(value_correction_shares, index_in_block,
                                    static_cast<uint8_t>(beta), invert);
      break;
    case 16:
      value_correction =
          ComputeValueCorrectionFor(value_correction_shares, index_in_block,
                                    static_cast<uint16_t>(beta), invert);
      break;
    case 32:
      value_correction =
          ComputeValueCorrectionFor(value_correction_shares, index_in_block,
                                    static_cast<uint32_t>(beta), invert);
      break;
    case 64:
      value_correction =
          ComputeValueCorrectionFor(value_correction_shares, index_in_block,
                                    static_cast<uint64_t>(beta), invert);
      break;
    case 128:
      value_correction =
          ComputeValueCorrectionFor(value_correction_shares, index_in_block,
                                    static_cast<absl::uint128>(beta), invert);
      break;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "`element_bitsize = ", element_bitsize, "` unimplemented"));
  }
  return value_correction;
}

// Expands the PRG seeds at the next `tree_level`, updates `seeds` and
// `control_bits`, and writes the next correction word to `keys`.
absl::Status DistributedPointFunction::GenerateNext(
    int tree_level, absl::uint128 alpha, absl::Span<const absl::uint128> beta,
    absl::Span<absl::uint128> seeds, absl::Span<bool> control_bits,
    absl::Span<DpfKey> keys) const {
  // As in `GenerateKeysIncremental`, we annotate code with the corresponding
  // lines from https://arxiv.org/pdf/2012.14884.pdf#figure.caption.12.
  //
  // Lines 13 & 14: Compute value correction word if there is a value on the
  // current level. This is done here already, since we use the "PRG evaluation
  // optimization" described in Appendix C.2 of the paper. Since we are using
  // fixed-key AES as PRG, which can have arbitrary stretch, this optimization
  // works even for large output groups.
  CorrectionWord* correction_word = keys[0].add_correction_words();
  if (tree_to_hierarchy_.contains(tree_level - 1)) {
    int hierarchy_level = tree_to_hierarchy_.at(tree_level - 1);
    absl::uint128 alpha_prefix = 0;
    int shift_amount = parameters_.back().log_domain_size() -
                       parameters_[hierarchy_level].log_domain_size();
    if (shift_amount < 128) {
      alpha_prefix = alpha >> shift_amount;
    }
    DPF_ASSIGN_OR_RETURN(
        absl::uint128 value_correction,
        ComputeValueCorrection(hierarchy_level, seeds, alpha_prefix,
                               beta[hierarchy_level], control_bits[1]));
    correction_word->mutable_output()->set_high(
        absl::Uint128High64(value_correction));
    correction_word->mutable_output()->set_low(
        absl::Uint128Low64(value_correction));
  }

  // Line 5: Expand seeds from previous level.
  std::array<std::array<absl::uint128, 2>, 2> expanded_seeds;
  DPF_RETURN_IF_ERROR(
      prg_left_.Evaluate(seeds, absl::MakeSpan(expanded_seeds[0])));
  DPF_RETURN_IF_ERROR(
      prg_right_.Evaluate(seeds, absl::MakeSpan(expanded_seeds[1])));
  std::array<std::array<bool, 2>, 2> expanded_control_bits;
  expanded_control_bits[0][0] = ExtractAndClearLowestBit(expanded_seeds[0][0]);
  expanded_control_bits[0][1] = ExtractAndClearLowestBit(expanded_seeds[0][1]);
  expanded_control_bits[1][0] = ExtractAndClearLowestBit(expanded_seeds[1][0]);
  expanded_control_bits[1][1] = ExtractAndClearLowestBit(expanded_seeds[1][1]);

  // Lines 6-8: Assign keep/lose branch depending on current bit of `alpha`.
  bool current_bit = 0;
  if (parameters_.back().log_domain_size() - tree_level < 128) {
    current_bit =
        (alpha & (absl::uint128{1}
                  << (parameters_.back().log_domain_size() - tree_level))) != 0;
  }
  bool keep = current_bit, lose = !current_bit;

  // Line 9: Compute seed correction word.
  absl::uint128 seed_correction =
      expanded_seeds[lose][0] ^ expanded_seeds[lose][1];

  // Line 10: Compute control bit correction words.
  std::array<bool, 2> control_bit_correction;
  control_bit_correction[0] = expanded_control_bits[0][0] ^
                              expanded_control_bits[0][1] ^ current_bit ^ 1;
  control_bit_correction[1] =
      expanded_control_bits[1][0] ^ expanded_control_bits[1][1] ^ current_bit;

  // We swap lines 11 and 12, since we first need to use the previous level's
  // control bits before updating them.

  // Line 12: Update seeds. Note that there is a typo in the paper: The
  // multiplication / AND needs to be done with the control bit of iteration
  // l-1, not l. Note that unlike the original algorithm, we are using the
  // corrected seed directly for the next iteration. This is secure as we're
  // using AES with a different key (kPrgKeyValue) to compute the value
  // correction word below.
  seeds[0] = expanded_seeds[keep][0];
  seeds[1] = expanded_seeds[keep][1];
  if (control_bits[0]) {
    seeds[0] ^= seed_correction;
  }
  if (control_bits[1]) {
    seeds[1] ^= seed_correction;
  }

  // Line 11: Update control bits.  Again, same typo as in Line 12.
  control_bits[0] = expanded_control_bits[keep][0] ^
                    (control_bits[0] & control_bit_correction[keep]);
  control_bits[1] = expanded_control_bits[keep][1] ^
                    (control_bits[1] & control_bit_correction[keep]);

  // Line 15: Assemble correction word and add it to keys[0].
  correction_word->mutable_seed()->set_high(
      absl::Uint128High64(seed_correction));
  correction_word->mutable_seed()->set_low(absl::Uint128Low64(seed_correction));
  correction_word->set_control_left(control_bit_correction[0]);
  correction_word->set_control_right(control_bit_correction[1]);

  // Copy correction word to second key.
  *(keys[1].add_correction_words()) = *correction_word;

  return absl::OkStatus();
}

absl::Status DistributedPointFunction::CheckContextParameters(
    const EvaluationContext& ctx) const {
  if (ctx.parameters_size() != static_cast<int>(parameters_.size())) {
    return absl::InvalidArgumentError(
        "Number of parameters in `ctx` doesn't match");
  }
  if (ctx.hierarchy_level() >= ctx.parameters_size()) {
    return absl::InvalidArgumentError(
        "This context has already been fully evaluated");
  }
  for (int i = 0; i < ctx.parameters_size(); ++i) {
    if (!google::protobuf::util::MessageDifferencer::Equivalent(
            ctx.parameters(i), parameters_[i])) {
      return absl::InvalidArgumentError(
          absl::StrCat("Parameter ", i, " in `ctx` doesn't match"));
    }
  }
  return absl::OkStatus();
}

absl::uint128 DistributedPointFunction::DomainToTreeIndex(
    absl::uint128 domain_index, int hierarchy_level) const {
  int block_index_bits = parameters_[hierarchy_level].log_domain_size() -
                         hierarchy_to_tree_[hierarchy_level];
  DCHECK(block_index_bits < 128);
  return domain_index >> block_index_bits;
}

int DistributedPointFunction::DomainToBlockIndex(absl::uint128 domain_index,
                                                 int hierarchy_level) const {
  int block_index_bits = parameters_[hierarchy_level].log_domain_size() -
                         hierarchy_to_tree_[hierarchy_level];
  DCHECK(block_index_bits < 128);
  return static_cast<int>(domain_index &
                          ((absl::uint128{1} << block_index_bits) - 1));
}

absl::StatusOr<DistributedPointFunction::DpfExpansion>
DistributedPointFunction::ExpandSeeds(
    const DpfExpansion& partial_evaluations,
    absl::Span<const CorrectionWord* const> correction_words) const {
  int num_expansions = static_cast<int>(correction_words.size());

  // Allocate buffers with the correct size to avoid reallocations.
  auto current_level_size =
      static_cast<int64_t>(partial_evaluations.seeds.size());
  DCHECK(num_expansions < 63);
  int64_t output_size = current_level_size << num_expansions;
  std::vector<absl::uint128> prg_buffer_left, prg_buffer_right;
  prg_buffer_left.reserve(output_size / 2);
  prg_buffer_right.reserve(output_size / 2);

  // Copy seeds and control bits. We will update these in-place.
  DpfExpansion expansion = partial_evaluations;
  expansion.seeds.reserve(output_size);
  expansion.control_bits.reserve(output_size);

  // We use an iterative expansion here to pipeline AES as much as possible.
  for (int i = 0; i < num_expansions; ++i) {
    // Expand PRG.
    prg_buffer_left.resize(current_level_size);
    prg_buffer_right.resize(current_level_size);
    DPF_RETURN_IF_ERROR(prg_left_.Evaluate(
        absl::MakeConstSpan(expansion.seeds).subspan(0, current_level_size),
        absl::MakeSpan(prg_buffer_left)));
    DPF_RETURN_IF_ERROR(prg_right_.Evaluate(
        absl::MakeConstSpan(expansion.seeds).subspan(0, current_level_size),
        absl::MakeSpan(prg_buffer_right)));

    // Merge results into next level of seeds and perform correction.
    absl::uint128 correction_seed = absl::MakeUint128(
        correction_words[i]->seed().high(), correction_words[i]->seed().low());
    bool correction_control_left = correction_words[i]->control_left();
    bool correction_control_right = correction_words[i]->control_right();
    expansion.seeds.resize(2 * current_level_size);
    expansion.control_bits.resize(2 * current_level_size);
    for (int64_t j = 0; j < current_level_size; ++j) {
      if (expansion.control_bits[j]) {
        prg_buffer_left[j] ^= correction_seed;
        prg_buffer_left[j] ^= correction_control_left;
        prg_buffer_right[j] ^= correction_seed;
        prg_buffer_right[j] ^= correction_control_right;
      }
      expansion.seeds[2 * j] = prg_buffer_left[j];
      expansion.seeds[2 * j + 1] = prg_buffer_right[j];
    }

    // Extract next level control bits and update size.
    for (int64_t j = 0; j < 2 * current_level_size; ++j) {
      expansion.control_bits[j] = ExtractAndClearLowestBit(expansion.seeds[j]);
    }
    current_level_size *= 2;
  }
  return expansion;
}

absl::StatusOr<DistributedPointFunction::DpfExpansion>
DistributedPointFunction::ExpandAndUpdateContext(
    absl::Span<const absl::uint128> prefixes, EvaluationContext& ctx) const {
  // Expand seeds by expanding either the DPF key seed, or
  // `ctx.partial_evaluations` for the given `prefixes`.
  DpfExpansion selected_partial_evaluations;
  int start_level = 0;
  if (ctx.hierarchy_level() == 0) {
    // First expansion -> Expand seed of the DPF key.
    selected_partial_evaluations.seeds = {
        absl::MakeUint128(ctx.key().seed().high(), ctx.key().seed().low())};
    selected_partial_evaluations.control_bits = {
        static_cast<bool>(ctx.key().party())};
  } else {
    // Second or later expansion -> Extract all seeds for `prefixes` from
    // `ctx.partial_evaluations`. To do that, Parse `ctx.partial_evaluations`
    // into another hash map for quick lookups up by prefix...
    absl::flat_hash_map<absl::uint128, std::pair<absl::uint128, bool>>
        partial_evaluations;
    for (const PartialEvaluation& element : ctx.partial_evaluations()) {
      absl::uint128 prefix =
          absl::MakeUint128(element.prefix().high(), element.prefix().low());
      // Try inserting `(seed, control_bit)` at `prefix` into
      // partial_evaluations. Return an error if `prefix` is already present.
      bool was_inserted;
      std::tie(std::ignore, was_inserted) =
          partial_evaluations.insert(std::make_pair(
              prefix, std::make_pair(absl::MakeUint128(element.seed().high(),
                                                       element.seed().low()),
                                     element.control_bit())));
      if (!was_inserted) {
        return absl::InvalidArgumentError(
            "Duplicate prefix in `ctx.partial_evaluations()`");
      }
    }
    // ... Then, select all partial evaluations from the hash map that
    // correspond to `prefixes`.
    selected_partial_evaluations.seeds.reserve(prefixes.size());
    selected_partial_evaluations.control_bits.reserve(prefixes.size());
    for (int64_t i = 0; i < static_cast<int64_t>(prefixes.size()); ++i) {
      auto it = partial_evaluations.find(prefixes[i]);
      if (it == partial_evaluations.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Prefix not present in ctx.partial_evaluations at hierarchy level ",
            ctx.hierarchy_level()));
      }
      std::pair<absl::uint128, bool> partial_evaluation = it->second;
      selected_partial_evaluations.seeds.push_back(partial_evaluation.first);
      selected_partial_evaluations.control_bits.push_back(
          partial_evaluation.second);
    }
    start_level = hierarchy_to_tree_[ctx.hierarchy_level() - 1];
  }

  // Expand up to the next hierarchy level.
  int stop_level = hierarchy_to_tree_[ctx.hierarchy_level()];
  DPF_ASSIGN_OR_RETURN(
      DpfExpansion expansion,
      ExpandSeeds(selected_partial_evaluations,
                  absl::MakeConstSpan(ctx.key().correction_words())
                      .subspan(start_level, stop_level - start_level)));

  // Update `partial_evaluations` in `ctx` if there are more evaluations to
  // come, and update `hierarchy_level`.
  ctx.clear_partial_evaluations();
  ctx.mutable_partial_evaluations()->Reserve(expansion.seeds.size());
  if (ctx.hierarchy_level() < static_cast<int>(parameters_.size()) - 1) {
    DCHECK(stop_level - start_level < 63);
    int64_t expanded_blocks_per_seed = 1LL << (stop_level - start_level);
    for (int64_t i = 0; i < static_cast<int64_t>(expansion.seeds.size()); ++i) {
      absl::uint128 current_prefix = 0;
      if (!prefixes.empty()) {
        current_prefix = prefixes[i / expanded_blocks_per_seed];
      }
      current_prefix = (current_prefix << (stop_level - start_level)) +
                       (i % expanded_blocks_per_seed);
      PartialEvaluation* current_element = ctx.add_partial_evaluations();
      current_element->mutable_prefix()->set_high(
          absl::Uint128High64(current_prefix));
      current_element->mutable_prefix()->set_low(
          absl::Uint128Low64(current_prefix));
      current_element->mutable_seed()->set_high(
          absl::Uint128High64(expansion.seeds[i]));
      current_element->mutable_seed()->set_low(
          absl::Uint128Low64(expansion.seeds[i]));
      current_element->set_control_bit(expansion.control_bits[i]);
    }
  }
  ctx.set_hierarchy_level(ctx.hierarchy_level() + 1);
  return expansion;
}

absl::StatusOr<std::unique_ptr<DistributedPointFunction>>
DistributedPointFunction::Create(const DpfParameters& parameters) {
  return CreateIncremental(absl::MakeConstSpan(&parameters, 1));
}

absl::StatusOr<std::unique_ptr<DistributedPointFunction>>
DistributedPointFunction::CreateIncremental(
    absl::Span<const DpfParameters> parameters) {
  // Check that parameters are valid.
  if (parameters.empty()) {
    return absl::InvalidArgumentError("`parameters` must not be empty");
  }
  // Sentinel values for checking that domain sizes are increasing and not too
  // far apart, and element sizes are non-decreasing.
  int previous_log_domain_size = 0;
  int previous_element_bitsize = 1;
  for (int i = 0; i < static_cast<int>(parameters.size()); ++i) {
    // Check log_domain_size.
    int32_t log_domain_size = parameters[i].log_domain_size();
    if (log_domain_size < 0) {
      return absl::InvalidArgumentError(
          "`log_domain_size` must be non-negative");
    }
    if (i > 0 && log_domain_size <= previous_log_domain_size) {
      return absl::InvalidArgumentError(
          "`log_domain_size` fields must be in ascending order in "
          "`parameters`");
    }
    // For full evaluation of a particular hierarchy level, want to be able to
    // represent 1 << (log_domain_size - previous_log_domain_size) in an
    // int64_t, so hierarchy levels may be at most 62 apart. Note that such
    // large gaps between levels are rare in practice, and in any case this
    // error can circumvented by adding additional intermediate hierarchy
    // levels.
    if (log_domain_size > previous_log_domain_size + 62) {
      return absl::InvalidArgumentError(
          "Hierarchies may be at most 62 levels apart");
    }
    previous_log_domain_size = log_domain_size;

    // Check element_bitsize.
    int32_t element_bitsize = parameters[i].element_bitsize();
    if (element_bitsize < 1) {
      return absl::InvalidArgumentError("`element_bitsize` must be positive");
    }
    if (element_bitsize > 128) {
      return absl::InvalidArgumentError(
          "`element_bitsize` must be less than or equal to 128");
    }
    if ((element_bitsize & (element_bitsize - 1)) != 0) {
      return absl::InvalidArgumentError(
          "`element_bitsize` must be a power of 2");
    }
    if (element_bitsize < previous_element_bitsize) {
      return absl::InvalidArgumentError(
          "`element_bitsize` fields must be non-decreasing in "
          "`parameters`");
    }
    previous_element_bitsize = element_bitsize;
  }

  // Map hierarchy levels to levels in the evaluation tree for value correction,
  // and vice versa.
  absl::flat_hash_map<int, int> tree_to_hierarchy;
  std::vector<int> hierarchy_to_tree(parameters.size());
  // Also keep track of the height needed for the evaluation tree.
  int tree_levels_needed = 1;
  for (int i = 0; i < static_cast<int>(parameters.size()); ++i) {
    int log_element_size =
        static_cast<int>(std::log2(parameters[i].element_bitsize()));
    // The tree level depends on the domain size and the element size. A single
    // AES block can fit 128 = 2^7 bits, so tree_level == log_domain_size iff
    // log_element_size == 7.
    int tree_level =
        std::max(0, parameters[i].log_domain_size() - 7 + log_element_size);
    tree_to_hierarchy[tree_level] = i;
    hierarchy_to_tree[i] = tree_level;
    tree_levels_needed = std::max(tree_levels_needed, tree_level + 1);
  }

  // Set up PRGs.
  DPF_ASSIGN_OR_RETURN(
      dpf_internal::PseudorandomGenerator prg_left,
      dpf_internal::PseudorandomGenerator::Create(kPrgKeyLeft));
  DPF_ASSIGN_OR_RETURN(
      dpf_internal::PseudorandomGenerator prg_right,
      dpf_internal::PseudorandomGenerator::Create(kPrgKeyRight));
  DPF_ASSIGN_OR_RETURN(
      dpf_internal::PseudorandomGenerator prg_value,
      dpf_internal::PseudorandomGenerator::Create(kPrgKeyValue));

  // Copy parameters and return new DPF.
  return absl::WrapUnique(new DistributedPointFunction(
      std::vector<DpfParameters>(parameters.begin(), parameters.end()),
      tree_levels_needed, std::move(tree_to_hierarchy),
      std::move(hierarchy_to_tree), std::move(prg_left), std::move(prg_right),
      std::move(prg_value)));
}

absl::StatusOr<std::pair<DpfKey, DpfKey>>
DistributedPointFunction::GenerateKeys(absl::uint128 alpha,
                                       absl::uint128 beta) const {
  return GenerateKeysIncremental(alpha, absl::MakeConstSpan(&beta, 1));
}

absl::StatusOr<std::pair<DpfKey, DpfKey>>
DistributedPointFunction::GenerateKeysIncremental(
    absl::uint128 alpha, absl::Span<const absl::uint128> beta) const {
  // Check validity of beta.
  if (beta.size() != parameters_.size()) {
    return absl::InvalidArgumentError(
        "`beta` has to have the same size as `parameters` passed at "
        "construction");
  }
  for (int i = 0; i < static_cast<int>(parameters_.size()); ++i) {
    if (parameters_[i].element_bitsize() < 128 &&
        beta[i] >= (absl::uint128{1} << (parameters_[i].element_bitsize()))) {
      return absl::InvalidArgumentError(
          absl::StrCat("`beta[", i, "]` larger than `parameters[", i,
                       "].element_bitsize()` allows"));
    }
  }

  // Check validity of alpha.
  int last_level_log_domain_size = parameters_.back().log_domain_size();
  if (last_level_log_domain_size < 128 &&
      alpha >= (absl::uint128{1} << last_level_log_domain_size)) {
    return absl::InvalidArgumentError(
        "`alpha` must be smaller than the output domain size");
  }

  std::array<DpfKey, 2> keys;
  keys[0].set_party(0);
  keys[1].set_party(1);

  // We will annotate the following code with the corresponding lines from the
  // pseudocode in the Incremental DPF paper
  // (https://arxiv.org/pdf/2012.14884.pdf, Figure 11).
  //
  // There are two possible dimensions for each variable at each level: Parties
  // (0 or 1) and branches (left or right). For two-dimensional arrays, we use
  // the outer dimension for the branch, and the inner dimension for the party.
  //
  // Line 2: Sample random seeds for each party.
  std::array<absl::uint128, 2> seeds;
  RAND_bytes(reinterpret_cast<uint8_t*>(&seeds[0]), sizeof(absl::uint128));
  RAND_bytes(reinterpret_cast<uint8_t*>(&seeds[1]), sizeof(absl::uint128));
  keys[0].mutable_seed()->set_high(absl::Uint128High64(seeds[0]));
  keys[0].mutable_seed()->set_low(absl::Uint128Low64(seeds[0]));
  keys[1].mutable_seed()->set_high(absl::Uint128High64(seeds[1]));
  keys[1].mutable_seed()->set_low(absl::Uint128Low64(seeds[1]));

  // Line 3: Initialize control bits.
  std::array<bool, 2> control_bits{0, 1};

  // Line 4: Compute correction words for each level after the first one.
  keys[0].mutable_correction_words()->Reserve(tree_levels_needed_ - 1);
  for (int i = 1; i < tree_levels_needed_; i++) {
    DPF_RETURN_IF_ERROR(GenerateNext(i, alpha, beta, absl::MakeSpan(seeds),
                                     absl::MakeSpan(control_bits),
                                     absl::MakeSpan(keys)));
  }

  // Compute output correction word for last layer.
  DPF_ASSIGN_OR_RETURN(
      absl::uint128 last_level_output_correction,
      ComputeValueCorrection(parameters_.size() - 1, seeds, alpha, beta.back(),
                             control_bits[1]));
  keys[0].mutable_last_level_output_correction()->set_high(
      absl::Uint128High64(last_level_output_correction));
  keys[0].mutable_last_level_output_correction()->set_low(
      absl::Uint128Low64(last_level_output_correction));
  keys[1].mutable_last_level_output_correction()->set_high(
      absl::Uint128High64(last_level_output_correction));
  keys[1].mutable_last_level_output_correction()->set_low(
      absl::Uint128Low64(last_level_output_correction));

  return std::make_pair(std::move(keys[0]), std::move(keys[1]));
}

absl::StatusOr<EvaluationContext>
DistributedPointFunction::CreateEvaluationContext(DpfKey key) {
  // Check that `key` is valid for this DPF.
  if (key.correction_words_size() != tree_levels_needed_ - 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Malformed DpfKey: expected ", tree_levels_needed_ - 1,
        " correction words, but got ", key.correction_words_size()));
  }
  for (int i = 0; i < static_cast<int>(hierarchy_to_tree_.size()); ++i) {
    if (hierarchy_to_tree_[i] == tree_levels_needed_ - 1) {
      // The output correction of the last tree level is always stored in
      // last_level_output_correction.
      continue;
    }
    if (!key.correction_words(hierarchy_to_tree_[i]).has_output()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Malformed DpfKey: expected correction_words[", hierarchy_to_tree_[i],
          "] to contain the output correction of hierarchy level ", i));
    }
  }
  // Create new EvaluationContext with `parameters_` and `key`.
  EvaluationContext result;
  for (int i = 0; i < static_cast<int>(parameters_.size()); ++i) {
    *(result.add_parameters()) = parameters_[i];
  }
  *(result.mutable_key()) = std::move(key);
  return result;
}

template <typename T>
absl::StatusOr<std::vector<T>> DistributedPointFunction::EvaluateNext(
    absl::Span<const absl::uint128> prefixes, EvaluationContext& ctx) const {
  DPF_RETURN_IF_ERROR(CheckContextParameters(ctx));
  if (prefixes.empty() && ctx.hierarchy_level() != 0) {
    return absl::InvalidArgumentError(
        "`prefixes` may only be empty if `ctx.hierarchy_level() == 0`");
  }
  if (sizeof(T) * 8 != parameters_[ctx.hierarchy_level()].element_bitsize()) {
    return absl::InvalidArgumentError(
        "Size of template parameter T doesn't match the element size of "
        "`ctx.hierarchy_level()`");
  }
  int current_hierarchy_level = ctx.hierarchy_level();
  int previous_log_domain_size = 0;
  if (current_hierarchy_level > 0) {
    previous_log_domain_size =
        parameters_[current_hierarchy_level - 1].log_domain_size();
    for (absl::uint128 prefix : prefixes) {
      if (previous_log_domain_size < 128 &&
          prefix > (absl::uint128{1} << previous_log_domain_size)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Index %d out of range for hierarchy level %d",
                            prefix, current_hierarchy_level));
      }
    }
  }
  int64_t prefixes_size = static_cast<int64_t>(prefixes.size());

  // The `prefixes` passed in by the caller refer to the domain of the previous
  // hierarchy level. However, because we batch multiple elements of type T in a
  // single uint128 block, multiple prefixes can actually refer to the same
  // block in the FSS evaluation tree. On a high level, our approach is as
  // follows:
  //
  // 1. Split up each element of `prefixes` into a tree index, pointing to a
  //    block in the FSS tree, and a block index, pointing to an element of type
  //    T in that block.
  //
  // 2. Compute a list of unique `tree_indices`, and for each original prefix,
  //    remember the position of the corresponding tree index in `tree_indices`.
  //
  // 3. After expanding the unique `tree_indices`, use the positions saved in
  //    Step (2) together with the corresponding block index to retrieve the
  //    expanded values for each prefix, and return them in the same order as
  //    `prefixes`.
  //
  // `tree_indices` holds the unique tree indices from `prefixes`, to be passed
  // to `ExpandAndUpdateContext`.
  std::vector<absl::uint128> tree_indices;
  tree_indices.reserve(prefixes_size);
  // `tree_indices_inverse` is the inverse of `tree_indices`, used for
  // deduplicating and constructing `prefix_map`.
  absl::flat_hash_map<absl::uint128, int64_t> tree_indices_inverse;
  // `prefix_map` maps each i < prefixes.size() to an element of `tree_indices`
  // and a block index. Used to select which elements to return after the
  // expansion, to ensure the result is ordered the same way as `prefixes`.
  std::vector<std::pair<int64_t, int>> prefix_map;
  prefix_map.reserve(prefixes_size);
  for (int64_t i = 0; i < prefixes_size; ++i) {
    absl::uint128 tree_index =
        DomainToTreeIndex(prefixes[i], current_hierarchy_level - 1);
    int block_index =
        DomainToBlockIndex(prefixes[i], current_hierarchy_level - 1);

    // Check if `tree_index` already exists in `tree_indices`.
    decltype(tree_indices_inverse)::iterator it;
    bool was_inserted;
    std::tie(it, was_inserted) = tree_indices_inverse.insert(
        std::make_pair(tree_index, tree_indices.size()));
    if (was_inserted) {
      tree_indices.push_back(tree_index);
    }
    prefix_map.push_back(std::make_pair(it->second, block_index));
  }
  tree_indices.shrink_to_fit();
  tree_indices_inverse.clear();

  // Perform expansion of unique `tree_indices`.
  DPF_ASSIGN_OR_RETURN(DpfExpansion expansion,
                       ExpandAndUpdateContext(tree_indices, ctx));

  // Get output correction word from `ctx`.
  constexpr int elements_per_block = dpf_internal::ElementsPerBlock<T>();
  const Block* output_correction = nullptr;
  if (current_hierarchy_level < static_cast<int>(parameters_.size()) - 1) {
    output_correction =
        &(ctx.key()
              .correction_words(hierarchy_to_tree_[current_hierarchy_level])
              .output());
  } else {
    // Last level output correction is stored in an extra proto field, since we
    // have one less correction word than tree levels.
    output_correction = &(ctx.key().last_level_output_correction());
  }

  // Split output correction into elements of type T.
  std::array<T, elements_per_block> correction_ints =
      dpf_internal::Uint128ToArray<T>(absl::MakeUint128(
          output_correction->high(), output_correction->low()));

  // Compute output PRG value of expanded seeds using prg_ctx_value_.
  std::vector<absl::uint128> hashed_expansion(expansion.seeds.size());
  DPF_RETURN_IF_ERROR(
      prg_value_.Evaluate(expansion.seeds, absl::MakeSpan(hashed_expansion)));

  // Compute value corrections for each block in `expanded_seeds`.
  std::vector<T> corrected_expansion(hashed_expansion.size() *
                                     elements_per_block);
  for (int64_t i = 0; i < static_cast<int64_t>(hashed_expansion.size()); ++i) {
    std::array<T, elements_per_block> current_elements =
        dpf_internal::Uint128ToArray<T>(hashed_expansion[i]);
    for (int j = 0; j < elements_per_block; ++j) {
      if (expansion.control_bits[i]) {
        current_elements[j] += correction_ints[j];
      }
      if (ctx.key().party() == 1) {
        current_elements[j] = -current_elements[j];
      }
      corrected_expansion[i * elements_per_block + j] = current_elements[j];
    }
  }

  // Compute the number of outputs we will have. For each prefix, we will have a
  // full expansion from the previous heirarchy level to the current heirarchy
  // level.
  int log_domain_size = parameters_[current_hierarchy_level].log_domain_size();
  DCHECK(log_domain_size - previous_log_domain_size < 63);
  int64_t outputs_per_prefix = 1LL
                               << (log_domain_size - previous_log_domain_size);

  if (current_hierarchy_level == 0) {
    // If prefixes is empty (i.e., current_hierarchy_level == 0), just return
    // the expansion, after shrinking to the correct size.
    corrected_expansion.resize(outputs_per_prefix);
    return corrected_expansion;
  } else {
    // Otherwise, only return elements under `prefixes`.
    int blocks_per_tree_prefix = expansion.seeds.size() / tree_indices.size();
    std::vector<T> result(prefixes_size * outputs_per_prefix);
    for (int64_t i = 0; i < prefixes_size; ++i) {
      int64_t prefix_expansion_start =
          prefix_map[i].first * blocks_per_tree_prefix * elements_per_block +
          prefix_map[i].second * outputs_per_prefix;
      std::copy_n(&corrected_expansion[prefix_expansion_start],
                  outputs_per_prefix, &result[i * outputs_per_prefix]);
    }
    return result;
  }
}

// Template instantiations for all types we support.
template absl::StatusOr<std::vector<uint8_t>>
DistributedPointFunction::EvaluateNext(absl::Span<const absl::uint128> prefixes,
                                       EvaluationContext& ctx) const;
template absl::StatusOr<std::vector<uint16_t>>
DistributedPointFunction::EvaluateNext(absl::Span<const absl::uint128> prefixes,
                                       EvaluationContext& ctx) const;
template absl::StatusOr<std::vector<uint32_t>>
DistributedPointFunction::EvaluateNext(absl::Span<const absl::uint128> prefixes,
                                       EvaluationContext& ctx) const;
template absl::StatusOr<std::vector<uint64_t>>
DistributedPointFunction::EvaluateNext(absl::Span<const absl::uint128> prefixes,
                                       EvaluationContext& ctx) const;
template absl::StatusOr<std::vector<absl::uint128>>
DistributedPointFunction::EvaluateNext(absl::Span<const absl::uint128> prefixes,
                                       EvaluationContext& ctx) const;

}  // namespace dpf
}  // namespace private_statistics
