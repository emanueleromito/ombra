// Query expansion tree (decision 0010, T0.5). Independent of OpenFHE and CUDA types so the same
// tree can run with the CPU reference operations (ombra_expand.cpp) and, later, GPU operations.
//
// The input is one ciphertext whose plaintext polynomial is the symmetric
// p(X) = q_0 + sum_{j>=1} q_j (X^j + X^-j), i.e. coefficient j is q_j and coefficient N-j is -q_j.
// With L = ceil(log2(2d)) (2^L >= 2d, so j and N-j fall in different residues mod 2^L), at level l
// (l = 0 .. L-1) every node c with residue r (its coefficients sit at indices congruent to r mod
// 2^l) is split with t = N/2^l + 1:
//     child r            = c + tau_t(c)
//     child r + 2^l      = (c - tau_t(c)) * X^(-2^l)
// After L levels the node with residue r < d is the constant polynomial 2^L * q_r. Branches whose
// smallest leaf index is >= d are never created, so exactly d outputs come out, all at depth L.
// The factor 2^L is NOT removed here; the caller adjusts the scaling factor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ombra::core {

struct ExpansionStats {
    std::size_t depth = 0;                    // L
    std::size_t nodes = 0;                    // all nodes created, root and leaves included
    std::size_t splits = 0;                   // calls to Ops::split
    std::size_t automorphisms = 0;            // key-switched substitutions (one per split)
    std::vector<std::size_t> nodes_per_level; // size L + 1
};

inline std::size_t expansion_depth(std::size_t d) {
    std::size_t depth = 0;
    while ((std::size_t{1} << depth) < 2 * d) {
        ++depth;
    }
    return depth;
}

// Ops must provide:
//   std::pair<Ct, Ct> split(const Ct& c, std::size_t level, std::uint32_t galois,
//                           std::size_t shift, bool want_odd);
// returning (c + tau_galois(c), (c - tau_galois(c)) * X^(-shift)); the second is only needed when
// want_odd is true (otherwise it may be left default-constructed).
// Returns the d leaves in index order.
template <class Ct, class Ops>
std::vector<Ct> expand_tree(const Ct& root, std::size_t d, std::size_t ring_dim, Ops& ops,
                            ExpansionStats* stats = nullptr) {
    if (d == 0 || d > ring_dim / 2) {  // also guarantees 2^L <= N
        throw std::invalid_argument("expansion: d must be in [1, N/2]");
    }
    const std::size_t depth = expansion_depth(d);
    ExpansionStats local;
    local.depth = depth;
    local.nodes = 1;
    local.nodes_per_level.push_back(1);

    std::vector<std::pair<std::size_t, Ct>> level_nodes;  // (residue, ciphertext)
    level_nodes.emplace_back(0, root);
    for (std::size_t l = 0; l < depth; ++l) {
        const std::size_t step = std::size_t{1} << l;
        const auto galois = static_cast<std::uint32_t>(ring_dim / step + 1);
        std::vector<std::pair<std::size_t, Ct>> next;
        next.reserve(level_nodes.size() * 2);
        for (auto& [r, c] : level_nodes) {
            const bool want_odd = (r + step) < d;
            auto children = ops.split(c, l, galois, step, want_odd);
            ++local.splits;
            ++local.automorphisms;
            next.emplace_back(r, std::move(children.first));
            if (want_odd) {
                next.emplace_back(r + step, std::move(children.second));
            }
        }
        local.nodes += next.size();
        local.nodes_per_level.push_back(next.size());
        level_nodes = std::move(next);
    }

    std::vector<Ct> out(d);
    for (auto& [r, c] : level_nodes) {
        out.at(r) = std::move(c);
    }
    if (level_nodes.size() != d) {
        throw std::logic_error("expansion: leaf count differs from d");
    }
    if (stats != nullptr) {
        *stats = std::move(local);
    }
    return out;
}

}  // namespace ombra::core
