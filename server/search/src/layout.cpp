#include "ombra/search/layout.hpp"

#include <algorithm>

namespace ombra::search {

std::size_t QueryLayout::filter_leaves() const {
  std::size_t s = 0;
  for (auto v : cardinalities) s += v;
  return s;
}

std::size_t max_filter_fields(std::size_t first_mod_bits, std::size_t scale_bits) {
  if (first_mod_bits < scale_bits + 5) return 0;
  const std::size_t allowed = std::size_t{1} << (first_mod_bits - scale_bits - 1 - 3);
  // The largest magnitude is a tombstoned slot: |score - lambda * (F + 1)| <= 1 + lambda * (F + 1).
  if (allowed < 1 + 2 * static_cast<std::size_t>(kFilterLambda)) return 0;
  const std::size_t f = (allowed - 1) / static_cast<std::size_t>(kFilterLambda) - 1;
  return std::min(f, kMaxFilterFields);
}

QueryLayout make_layout(const storage::CollectionSpec& spec, std::size_t ring_dim,
                        std::size_t f_max) {
  if (spec.dimension == 0) throw LayoutError("layout: dimension must be positive");
  QueryLayout l;
  l.d = spec.dimension;
  if (spec.filter_fields.size() > std::min(kMaxFilterFields, f_max))
    throw LayoutError("layout: too many filter fields for this preset");
  std::uint32_t expect = 0;
  for (const auto& f : spec.filter_fields) {
    if (f.field_id != expect++) throw LayoutError("layout: filter field ids must be 0, 1, ...");
    if (f.cardinality < 1 || f.cardinality > kMaxFilterCardinality)
      throw LayoutError("layout: filter cardinality out of range");
    l.cardinalities.push_back(f.cardinality);
  }
  if (l.leaves() > ring_dim / 2) throw LayoutError("layout: d + sum(cardinality) exceeds N/2");
  return l;
}

std::vector<double> penalty_vector(const std::vector<std::string>& ids,
                                   const std::unordered_set<std::string>& tombstones,
                                   std::size_t slots, double penalty) {
  if (ids.size() > slots) throw LayoutError("penalty: more ids than slots");
  std::vector<double> v;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (!tombstones.count(ids[i])) continue;
    if (v.empty()) v.assign(slots, 0.0);
    v[i] = penalty;
  }
  return v;
}

}  // namespace ombra::search
