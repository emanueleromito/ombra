// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CPU unit test of the expansion tree shape (expansion_tree.hpp): expand_tree_into must hand the
// last level to the sink with exactly the leaves expand_tree returns, and fill the same statistics.
// A fake ciphertext (its own residue) stands for the real ones; the real GPU path is checked
// against OpenFHE in core/src/ombra_expand_gpu.cpp (stage 5).
#include "ombra/core/expansion_tree.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <utility>
#include <vector>

using namespace ombra::core;

namespace {

int failures = 0;

// Ct = the residue (leaf index) the node stands for; splitting node r gives r and r + shift.
struct FakeOps {
  using Ct = long;
  std::size_t into_calls = 0;
  std::size_t last_level = 0;

  std::vector<std::pair<Ct, Ct>> split_level(const std::vector<Ct>& nodes, std::size_t,
                                             std::uint32_t, std::size_t shift,
                                             const std::vector<char>& want_odd) {
    std::vector<std::pair<Ct, Ct>> out;
    for (std::size_t k = 0; k < nodes.size(); ++k) {
      out.emplace_back(nodes[k], want_odd[k] ? nodes[k] + static_cast<Ct>(shift) : -1);
    }
    return out;
  }

  struct Sink {
    std::vector<long>* leaves;
  };

  void split_level_into(const std::vector<Ct>& nodes, std::size_t level, std::uint32_t,
                        std::size_t shift, const std::vector<char>& want_odd,
                        const std::vector<std::size_t>& residues, const Sink& sink) {
    ++into_calls;
    last_level = level;
    for (std::size_t k = 0; k < nodes.size(); ++k) {
      if (static_cast<std::size_t>(nodes[k]) != residues[k]) throw std::logic_error("residue");
      (*sink.leaves).at(residues[k]) = nodes[k];
      if (want_odd[k]) (*sink.leaves).at(residues[k] + shift) = nodes[k] + static_cast<long>(shift);
    }
  }
};

void expect(bool ok, const char* what, std::size_t d) {
  if (!ok) {
    std::printf("FAIL %s (d=%zu)\n", what, d);
    ++failures;
  }
}

}  // namespace

int main() {
  const std::size_t ring = 4096;
  for (std::size_t d : {1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17, 100, 384, 768, 1536, 2048}) {
    FakeOps a, b;
    ExpansionStats sa, sb;
    const auto leaves = expand_tree<long>(0, d, ring, a, &sa);
    std::vector<long> into(d, -7);  // sentinel: every leaf must be written
    FakeOps::Sink sink{&into};
    expand_tree_into<long>(0, d, ring, b, sink, &sb);
    bool same = leaves.size() == d;
    for (std::size_t i = 0; same && i < d; ++i) same = leaves[i] == static_cast<long>(i);
    expect(same, "expand_tree leaves are 0..d-1", d);
    expect(into == leaves, "expand_tree_into leaves equal expand_tree leaves", d);
    expect(b.into_calls == 1 && b.last_level + 1 == expansion_depth(d),
           "sink called once, at the last level", d);
    expect(sa.depth == sb.depth && sa.nodes == sb.nodes && sa.splits == sb.splits &&
               sa.automorphisms == sb.automorphisms && sa.nodes_per_level == sb.nodes_per_level,
           "statistics equal", d);
  }
  // Refused sizes.
  for (std::size_t d : {std::size_t{0}, ring / 2 + 1}) {
    FakeOps b;
    std::vector<long> into(8);
    FakeOps::Sink sink{&into};
    bool threw = false;
    try {
      expand_tree_into<long>(0, d, ring, b, sink);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    expect(threw, "expand_tree_into refuses d outside [1, N/2]", d);
  }
  if (failures != 0) {
    std::printf("expansion tree test: FAIL (%d)\n", failures);
    return 1;
  }
  std::printf("expansion tree test: PASS\n");
  return 0;
}
