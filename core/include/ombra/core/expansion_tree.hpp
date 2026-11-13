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
// The factor 2^L is NOT removed here and no multiplication or scale change is done anywhere: the
// client pre-scales the query by 2^-L before encoding, so each leaf is the constant q_r.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
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
// Ops may additionally provide a level-at-once operation (T0.5b):
//   std::vector<std::pair<Ct, Ct>> split_level(const std::vector<Ct>& nodes, std::size_t level,
//                                              std::uint32_t galois, std::size_t shift,
//                                              const std::vector<char>& want_odd);
// which must return exactly what calling split on every node would. All nodes of a level share the
// Galois element and the shift, so a GPU can batch them. If it is absent, split is called per node.
// Returns the d leaves in index order.
//
// expand_tree_into (below) runs the same tree but never materialises the leaves as Ct objects: the
// last level is handed to  ops.split_level_into(nodes, level, galois, shift, want_odd, residues,
// sink)  which must write the even child of nodes[k] to leaf residues[k] and, if want_odd[k], the
// odd child to leaf residues[k] + shift, wherever `sink` says (e.g. one contiguous device buffer).
namespace detail {

// Shared level loop. Sink = std::nullptr_t: the d leaves are returned. Otherwise the last level
// goes to the sink through ops.split_level_into and an empty vector is returned.
template <class Ct, class Ops, class Sink>
std::vector<Ct> expand_tree_core(const Ct& root, std::size_t d, std::size_t ring_dim, Ops& ops,
                                 ExpansionStats* stats, const Sink& sink) {
    (void)sink;
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
        if constexpr (!std::is_null_pointer_v<Sink>) {
            if (l + 1 == depth) {  // last level: children go straight to the sink
                std::vector<Ct> nodes;
                std::vector<char> odd;
                std::vector<std::size_t> residues;
                nodes.reserve(level_nodes.size());
                std::size_t leaves = 0;
                for (auto& [r, c] : level_nodes) {
                    nodes.push_back(std::move(c));
                    odd.push_back((r + step) < d ? 1 : 0);
                    residues.push_back(r);
                    leaves += odd.back() ? 2 : 1;
                    ++local.splits;
                    ++local.automorphisms;
                }
                if (leaves != d) {
                    throw std::logic_error("expansion: leaf count differs from d");
                }
                ops.split_level_into(nodes, l, galois, step, odd, residues, sink);
                nodes.clear();  // release the parents' handles
                local.nodes += leaves;
                local.nodes_per_level.push_back(leaves);
                if (stats != nullptr) {
                    *stats = std::move(local);
                }
                return {};
            }
        }
        std::vector<std::pair<std::size_t, Ct>> next;
        next.reserve(level_nodes.size() * 2);
        if constexpr (requires(Ops& o, const std::vector<Ct>& v, const std::vector<char>& w) {
                          o.split_level(v, l, galois, step, w);
                      }) {
            std::vector<Ct> nodes;
            std::vector<char> odd;
            nodes.reserve(level_nodes.size());
            for (auto& [r, c] : level_nodes) {
                nodes.push_back(std::move(c));
                odd.push_back((r + step) < d ? 1 : 0);
            }
            auto children = ops.split_level(nodes, l, galois, step, odd);
            nodes.clear();  // release the parents' handles before the next level
            for (std::size_t k = 0; k < level_nodes.size(); ++k) {
                const std::size_t r = level_nodes[k].first;
                ++local.splits;
                ++local.automorphisms;
                next.emplace_back(r, std::move(children[k].first));
                if (odd[k]) {
                    next.emplace_back(r + step, std::move(children[k].second));
                }
            }
        } else {
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

}  // namespace detail

template <class Ct, class Ops>
std::vector<Ct> expand_tree(const Ct& root, std::size_t d, std::size_t ring_dim, Ops& ops,
                            ExpansionStats* stats = nullptr) {
    return detail::expand_tree_core(root, d, ring_dim, ops, stats, nullptr);
}

// Same tree, but the d leaves are written by ops.split_level_into into `sink` (see above) instead
// of being returned. d >= 1 always gives depth >= 1 (2^depth >= 2d >= 2), so the last level exists
// and every leaf, including d == 1, is produced by a split; the root alone is never a leaf.
template <class Ct, class Ops, class Sink>
void expand_tree_into(const Ct& root, std::size_t d, std::size_t ring_dim, Ops& ops,
                      const Sink& sink, ExpansionStats* stats = nullptr) {
    static_assert(!std::is_null_pointer_v<Sink>, "expand_tree_into needs a sink");
    if (d == 0 || d > ring_dim / 2 || expansion_depth(d) == 0) {
        throw std::invalid_argument("expansion: d must be in [1, N/2]");
    }
    detail::expand_tree_core(root, d, ring_dim, ops, stats, sink);
}

// Nodes alive at the same time while expanding ONE tree level by level: a level's parents are
// released only after the whole level is split, so the peak is the largest sum of two consecutive
// levels. Level l has min(2^l, d) nodes (residue r exists iff r < d). Used to plan device memory.
inline std::size_t expansion_peak_nodes(std::size_t d) {
    const std::size_t depth = expansion_depth(d);
    const auto count = [d](std::size_t l) { return std::min(std::size_t{1} << l, d); };
    std::size_t peak = 0;
    for (std::size_t l = 0; l < depth; ++l) {
        peak = std::max(peak, count(l) + count(l + 1));
    }
    return peak;
}

// Expands the trees of several queries together (T0.7). The tree shape depends only on d, so node
// k of every query is at the same level with the same residue: each level concatenates the nodes of
// all queries (query-major) into one split_level call, which the GPU runs in batched launches with
// the one Galois key and monomial of that level. Result: leaves[query][index], each exactly what
// expand_tree gives for that query's root. Every query's parents are released after its level is
// split. Ops should provide split_level; without it split is called per node. `stats` (optional)
// counts the splits and nodes of all queries together. expand_tree itself is unchanged.
template <class Ct, class Ops>
std::vector<std::vector<Ct>> expand_trees(const std::vector<Ct>& roots, std::size_t d,
                                          std::size_t ring_dim, Ops& ops,
                                          ExpansionStats* stats = nullptr) {
    if (d == 0 || d > ring_dim / 2) {
        throw std::invalid_argument("expansion: d must be in [1, N/2]");
    }
    if (roots.empty()) {
        throw std::invalid_argument("expansion: no queries");
    }
    const std::size_t queries = roots.size();
    const std::size_t depth = expansion_depth(d);
    ExpansionStats local;
    local.depth = depth;
    local.nodes = queries;
    local.nodes_per_level.push_back(queries);

    std::vector<std::size_t> residues{0};  // the same for every query
    std::vector<std::vector<Ct>> cts(queries);
    for (std::size_t q = 0; q < queries; ++q) {
        cts[q].push_back(roots[q]);
    }
    for (std::size_t l = 0; l < depth; ++l) {
        const std::size_t step = std::size_t{1} << l;
        const auto galois = static_cast<std::uint32_t>(ring_dim / step + 1);
        const std::size_t width = residues.size();
        std::vector<char> odd_one(width);
        for (std::size_t k = 0; k < width; ++k) {
            odd_one[k] = (residues[k] + step) < d ? 1 : 0;
        }
        std::vector<std::size_t> next_residues;
        for (std::size_t k = 0; k < width; ++k) {
            next_residues.push_back(residues[k]);
            if (odd_one[k]) {
                next_residues.push_back(residues[k] + step);
            }
        }
        std::vector<std::vector<Ct>> next(queries);
        if constexpr (requires(Ops& o, const std::vector<Ct>& v, const std::vector<char>& w) {
                          o.split_level(v, l, galois, step, w);
                      }) {
            std::vector<Ct> nodes;
            std::vector<char> odd;
            nodes.reserve(queries * width);
            odd.reserve(queries * width);
            for (std::size_t q = 0; q < queries; ++q) {
                for (std::size_t k = 0; k < width; ++k) {
                    nodes.push_back(std::move(cts[q][k]));
                    odd.push_back(odd_one[k]);
                }
                cts[q].clear();
            }
            auto children = ops.split_level(nodes, l, galois, step, odd);
            nodes.clear();  // release the parents' handles before the next level
            for (std::size_t q = 0; q < queries; ++q) {
                next[q].reserve(next_residues.size());
                for (std::size_t k = 0; k < width; ++k) {
                    auto& ch = children[q * width + k];
                    ++local.splits;
                    ++local.automorphisms;
                    next[q].push_back(std::move(ch.first));
                    if (odd_one[k]) {
                        next[q].push_back(std::move(ch.second));
                    }
                }
            }
        } else {
            for (std::size_t q = 0; q < queries; ++q) {
                for (std::size_t k = 0; k < width; ++k) {
                    auto ch = ops.split(cts[q][k], l, galois, step, odd_one[k] != 0);
                    ++local.splits;
                    ++local.automorphisms;
                    next[q].push_back(std::move(ch.first));
                    if (odd_one[k]) {
                        next[q].push_back(std::move(ch.second));
                    }
                }
                cts[q].clear();
            }
        }
        cts = std::move(next);
        residues = std::move(next_residues);
        local.nodes += residues.size() * queries;
        local.nodes_per_level.push_back(residues.size() * queries);
    }

    if (residues.size() != d) {
        throw std::logic_error("expansion: leaf count differs from d");
    }
    std::vector<std::vector<Ct>> out(queries, std::vector<Ct>(d));
    for (std::size_t q = 0; q < queries; ++q) {
        for (std::size_t k = 0; k < d; ++k) {
            out[q].at(residues[k]) = std::move(cts[q][k]);
        }
    }
    if (stats != nullptr) {
        *stats = std::move(local);
    }
    return out;
}

}  // namespace ombra::core
