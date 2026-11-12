// Pure logic of the search path, no OpenFHE and no CUDA: the query/column layout of a
// collection (decision 0019, additive penalty) and the tombstone penalty vector.
//
// A collection has d vector columns and F filter fields with cardinalities V_1..V_F. A data block
// has D = d + sum(V) columns: the d vector columns first, then the one-hot columns field by field
// (field-major, as in the proto). The expanded query has the same D leaves; the client has already
// scaled the filter leaves by lambda = 4. A cluster-center block has d columns only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "ombra/storage/catalog.hpp"

namespace ombra::search {

// The weight of one matching filter field in a decrypted slot (decision 0019).
inline constexpr double kFilterLambda = 4.0;
// Proto limits (LIMIT_MAX_FILTER_FIELDS, LIMIT_MAX_FILTER_CARDINALITY).
inline constexpr std::size_t kMaxFilterFields = 16;
inline constexpr std::size_t kMaxFilterCardinality = 64;

class LayoutError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

struct QueryLayout {
  std::size_t d = 0;
  std::vector<std::uint32_t> cardinalities;  // per filter field, in field order

  std::size_t fields() const { return cardinalities.size(); }
  std::size_t filter_leaves() const;
  std::size_t leaves() const { return d + filter_leaves(); }  // D
  std::size_t data_columns() const { return leaves(); }
  std::size_t center_columns() const { return d; }
  // Added to a tombstoned slot: -lambda * (F + 1). With c in [0, F] a tombstoned slot decrypts to
  // at most 1 + lambda*F - lambda*(F+1) = 1 - lambda, below every live slot (at least -1).
  double tombstone_penalty() const { return -kFilterLambda * static_cast<double>(fields() + 1); }
};

// Largest F the headroom of a preset allows (decision 0019): a slot holds at most
// 1 + lambda * F and must stay below 2^(first_mod_bits - scale_bits - 1 - 3), also capped at
// kMaxFilterFields. Returns 0 if even F = 1 does not fit.
std::size_t max_filter_fields(std::size_t first_mod_bits, std::size_t scale_bits);

// Builds the layout of a collection and checks it: d > 0, F <= min(kMaxFilterFields, f_max), each
// cardinality in 1..kMaxFilterCardinality, and D <= ring_dim / 2 (the expansion tree separates
// index j from N - j). Throws LayoutError.
QueryLayout make_layout(const storage::CollectionSpec& spec, std::size_t ring_dim,
                        std::size_t f_max);

// Penalty values for the slots of a block: the vector has `slots` entries, `penalty` where the id
// at that slot is tombstoned and 0 elsewhere. Returns an empty vector if no id of the block is
// tombstoned (the block then needs no plaintext addition). Throws LayoutError if there are more
// ids than slots.
std::vector<double> penalty_vector(const std::vector<std::string>& ids,
                                   const std::unordered_set<std::string>& tombstones,
                                   std::size_t slots, double penalty);

}  // namespace ombra::search
