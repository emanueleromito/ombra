// ombra_search_engine_check: oracle driver for the search engine (T2.7a).
//
// Usage: ombra_search_engine_check <in_dir> <out_dir> <gpu_blocks> <host_blocks>
//
// <in_dir> holds public material written by tests/test_gpu_search_engine.py (never a secret key):
//   manifest.txt   preset, OpenFHE version, collection, key tag, d, filter cardinalities, cluster
//                  count K, one "block <cluster> <idx> <n_ids>" line per stored block (cluster K =
//                  the center blocks), "delete <id>" lines, and the query names
//   context.bin, evalmult.bin, autokeys.bin           OpenFHE BINARY serializations
//   col_<cluster>_<idx>_<j>.bin                       column j of a block (d + sum(V) columns, centers d)
//   query_<name>.bin, query_wrongkey.bin              packed queries of D entries
// Vector ids are "<cluster>.<idx>.<slot>".
//
// The driver builds a Store from these (real insert, delete and key paths), opens a SearchEngine
// whose GPU cache budget is gpu_blocks data blocks (below the collection size) and host pool
// host_blocks, and for every query runs stage 1 and stage 2 over all clusters, plus a repeated
// round on the same session. Every output ciphertext is compared EXACTLY (every RNS word, moduli,
// level, noise degree, scaling factor, key tag) with the OpenFHE CPU chain on the same inputs:
//   CPU expansion tree -> sum_j EvalMultNoRelin(column_j, leaf_j) -> Relinearize -> Rescale
//   -> EvalAdd(plaintext penalty) on blocks with a deleted id.
// It then checks that wrong key tags (request, ciphertext, key files), a missing key upload and a
// wrong filter_leaves are refused. Output envelopes go to <out_dir> for decryption by the test,
// which alone holds the secret key. Prints PASS/FAIL lines, counts and timings, never data.
// Exit code: 0 all exact, 1 mismatch, 2 error.

#include <typeinfo>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/file_envelope.hpp"
#include "ombra/search/session.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/store.hpp"
#include "openfhe.h"
#include "openfhe_expand_ops.hpp"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
namespace core = ombra::core;
namespace fs = std::filesystem;
using Ct = Ciphertext<DCRTPoly>;

namespace {

int g_failures = 0;

struct Manifest {
  std::string preset, version, collection, key_tag;
  std::size_t d = 0, clusters = 0;
  std::vector<std::uint32_t> cards;
  struct Block {
    std::uint32_t cluster, idx, n_ids;
  };
  std::vector<Block> blocks;
  std::vector<std::string> deleted, queries;
};

Manifest read_manifest(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read manifest");
  Manifest m;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream in(line);
    std::string k;
    in >> k;
    if (k == "preset") in >> m.preset;
    else if (k == "version") in >> m.version;
    else if (k == "collection") in >> m.collection;
    else if (k == "key_tag") in >> m.key_tag;
    else if (k == "d") in >> m.d;
    else if (k == "clusters") in >> m.clusters;
    else if (k == "card") for (std::uint32_t c; in >> c;) m.cards.push_back(c);
    else if (k == "block") {
      Manifest::Block b{};
      in >> b.cluster >> b.idx >> b.n_ids;
      m.blocks.push_back(b);
    } else if (k == "delete") {
      std::string id;
      in >> id;
      m.deleted.push_back(id);
    } else if (k == "query") {
      std::string n;
      in >> n;
      m.queries.push_back(n);
    }
  }
  return m;
}

std::string id_of(std::uint32_t cluster, std::uint32_t idx, std::size_t slot) {
  return std::to_string(cluster) + "." + std::to_string(idx) + "." + std::to_string(slot);
}

Ct load_ct(const std::string& path) {
  Ct ct;
  if (!Serial::DeserializeFromFile(path, ct, SerType::BINARY))
    throw std::runtime_error("cannot read ciphertext " + path);
  return ct;
}

Ct ct_from_envelope(const std::vector<std::uint8_t>& env, const Manifest& m) {
  const core::Envelope e =
      core::read_envelope(env, core::EnvelopeKind::Ciphertext, m.preset, m.version);
  std::istringstream in(std::string(e.payload.begin(), e.payload.end()),
                        std::ios::in | std::ios::binary);
  Ct ct;
  Serial::Deserialize(ct, in, SerType::BINARY);
  if (!ct) throw std::runtime_error("cannot read a result ciphertext");
  return ct;
}

std::vector<std::uint8_t> wrap(core::EnvelopeKind kind, const Manifest& m,
                               const std::vector<std::uint8_t>& payload) {
  return core::write_envelope({kind, m.preset, m.version, payload});
}

// Exact comparison; prints the first difference as indices only.
void check(const std::string& stage, const Ct& gpu, const Ct& cpu) {
  const auto a = core::flatten_one(gpu);
  const auto b = core::flatten_one(cpu);
  bool ok = true;
  if (a.components != b.components || a.num_towers != b.num_towers || a.ring_dim != b.ring_dim ||
      a.moduli != b.moduli) {
    std::printf("%s: FAIL (shape or moduli differ)\n", stage.c_str());
    ok = false;
  } else if (a.key_tag != b.key_tag || a.encoding != b.encoding) {
    std::printf("%s: FAIL (key tag or encoding type differs)\n", stage.c_str());
    ok = false;
  } else if (a.level != b.level || a.noise_scale_deg != b.noise_scale_deg ||
             a.scaling_factor != b.scaling_factor) {
    std::printf("%s: FAIL (level %zu/%zu, noise degree %zu/%zu, scaling factor %s)\n",
                stage.c_str(), a.level, b.level, a.noise_scale_deg, b.noise_scale_deg,
                a.scaling_factor == b.scaling_factor ? "equal" : "differs");
    ok = false;
  } else if (gpu->GetScalingFactorInt() != cpu->GetScalingFactorInt()) {
    std::printf("%s: FAIL (integer scaling factor differs)\n", stage.c_str());
    ok = false;
  } else if (a.words != b.words) {
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < a.words.size(); ++i)
      if (a.words[i] != b.words[i] && bad++ == 0) first = i;
    std::printf("%s: FAIL (%zu of %zu words differ; first at component %zu, tower %zu, index %zu)\n",
                stage.c_str(), bad, a.words.size(), first / (a.num_towers * a.ring_dim),
                (first / a.ring_dim) % a.num_towers, first % a.ring_dim);
    ok = false;
  }
  if (ok)
    std::printf("%s: PASS (level %zu, noise degree %zu, %zu components x %zu towers)\n",
                stage.c_str(), a.level, a.noise_scale_deg, a.components, a.num_towers);
  else
    ++g_failures;
}

void expect(const std::string& what, bool ok) {
  std::printf("%s: %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  if (!ok) ++g_failures;
}

template <class E, class F>
bool throws_as(F&& f) {
  try {
    f();
  } catch (const E&) {
    return true;
  } catch (const std::exception& ex) {
    // The text of our own errors carries no data; print the type to diagnose a wrong refusal.
    std::fprintf(stderr, "  threw %s: %s\n", typeid(ex).name(), ex.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "  threw a non-std exception\n");
    return false;
  }
  std::fprintf(stderr, "  did not throw\n");
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: ombra_search_engine_check <in_dir> <out_dir> <gpu_blocks> <host_blocks>\n");
    return 2;
  }
  try {
    const std::string in_dir = argv[1], out_dir = argv[2];
    const std::size_t gpu_blocks = std::stoul(argv[3]), host_blocks = std::stoul(argv[4]);
    const Manifest m = read_manifest(in_dir + "/manifest.txt");
    const std::size_t F = m.cards.size();
    std::size_t sum_v = 0;
    for (auto v : m.cards) sum_v += v;
    const std::size_t D = m.d + sum_v;
    const double penalty = -4.0 * static_cast<double>(F + 1);  // decision 0019, independent of the engine
    const std::set<std::string> deleted(m.deleted.begin(), m.deleted.end());

    // ---- Oracle context (public material only) ------------------------------------------------
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(in_dir + "/context.bin", cc, SerType::BINARY))
      throw std::runtime_error("cannot read crypto context");
    const std::size_t ring = cc->GetRingDimension();
    const std::size_t towers = cc->GetCryptoParameters()->GetElementParams()->GetParams().size();
    const std::size_t block_bytes = D * 2 * towers * ring * sizeof(std::uint64_t);

    // ---- Store: real insert, delete and key paths ---------------------------------------------
    const std::string store_dir = out_dir + "/store";
    fs::remove_all(store_dir);
    ombra::storage::Store store(store_dir, m.version);
    ombra::storage::CollectionSpec spec;
    spec.api_version = 1;
    spec.name = m.collection;
    spec.preset = m.preset;
    spec.dimension = static_cast<std::uint32_t>(m.d);
    spec.security_level = 1;
    for (std::uint32_t i = 0; i < m.cards.size(); ++i) spec.filter_fields.push_back({i, m.cards[i]});
    spec.key_tag = m.key_tag;
    spec.cluster_count = static_cast<std::uint32_t>(m.clusters);
    store.create_collection(spec);

    const auto file_bytes = [&](const std::string& n) {
      return ombra::storage::read_file(in_dir + "/" + n);
    };
    const auto put_keys = [&](const std::string& coll) {
      store.put_key_file(coll, ombra::storage::kKeyContext,
                         wrap(core::EnvelopeKind::Context, m, file_bytes("context.bin")));
      store.put_key_file(coll, ombra::storage::kKeyEvalMult,
                         wrap(core::EnvelopeKind::PublicEvalKey, m, file_bytes("evalmult.bin")));
      store.put_key_file(coll, ombra::storage::kKeyAutomorphism,
                         wrap(core::EnvelopeKind::PublicEvalKey, m, file_bytes("autokeys.bin")));
    };

    std::map<std::string, Manifest::Block> block_of_file;
    std::size_t collection_bytes = 0;
    for (const auto& b : m.blocks) {
      const std::size_t ncols = (b.cluster == m.clusters) ? m.d : D;
      std::vector<std::string> ids;
      for (std::size_t s = 0; s < b.n_ids; ++s) ids.push_back(id_of(b.cluster, b.idx, s));
      std::vector<std::vector<std::uint8_t>> cols;
      for (std::size_t j = 0; j < ncols; ++j)
        cols.push_back(wrap(core::EnvelopeKind::Ciphertext, m,
                            file_bytes("col_" + std::to_string(b.cluster) + "_" +
                                       std::to_string(b.idx) + "_" + std::to_string(j) + ".bin")));
      block_of_file[store.insert(m.collection, b.cluster, ids, cols)] = b;
      collection_bytes += ncols * 2 * towers * ring * sizeof(std::uint64_t);
    }
    store.remove(m.collection, m.deleted);

    // ---- Refusals before the keys are uploaded -------------------------------------------------
    ombra::search::EngineConfig cfg;
    cfg.gpu_budget_bytes = gpu_blocks * block_bytes;
    cfg.host_budget_bytes = host_blocks * block_bytes;
    cfg.lookahead = 1;
    cfg.blocks_per_batch = 3;  // ragged chunks: 4 blocks of a query give batches of 3 and 1
    ombra::search::SearchEngine engine(store, cfg);
    {
      const auto q = wrap(core::EnvelopeKind::Ciphertext, m, file_bytes("query_" + m.queries[0] + ".bin"));
      expect("search before key upload is refused (keys missing)",
             throws_as<ombra::search::KeysMissing>(
                 [&] { engine.open_session(m.collection, m.key_tag, q, sum_v); }));
      expect("loading a collection without keys is refused",
             throws_as<ombra::search::KeysMissing>([&] { engine.load_collection(m.collection); }));
    }
    put_keys(m.collection);
    engine.load_collection(m.collection);

    // A second collection whose spec names another key tag, with the same key files: refused.
    {
      auto other = spec;
      other.name = "other";
      other.key_tag = "not-the-key-tag-of-these-files";
      store.create_collection(other);
      put_keys("other");
      expect("key files made under another key than the collection's are refused",
             throws_as<ombra::search::KeyTagMismatch>([&] { engine.load_collection("other"); }));
    }

    const auto& akeys = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(m.key_tag);
    const auto envelope_of_query = [&](const std::string& name) {
      return wrap(core::EnvelopeKind::Ciphertext, m, file_bytes("query_" + name + ".bin"));
    };

    {
      const auto q = envelope_of_query(m.queries[0]);
      expect("request with a wrong key tag is refused",
             throws_as<ombra::search::KeyTagMismatch>(
                 [&] { engine.open_session(m.collection, "wrong-tag", q, sum_v); }));
      expect("query ciphertext made under another key is refused",
             throws_as<ombra::search::KeyTagMismatch>([&] {
               engine.open_session(m.collection, m.key_tag, envelope_of_query("wrongkey"), sum_v);
             }));
      expect("wrong filter_leaves is refused",
             throws_as<ombra::search::InvalidQuery>(
                 [&] { engine.open_session(m.collection, m.key_tag, q, sum_v + 1); }));
      auto bad = q;
      bad[4] = 9;  // unknown envelope version
      expect("query envelope with an unknown version is refused",
             throws_as<ombra::search::InvalidQuery>(
                 [&] { engine.open_session(m.collection, m.key_tag, bad, sum_v); }));
    }

    // ---- Oracle chain for one block ------------------------------------------------------------
    const auto cpu_chain = [&](const Manifest::Block& b, const std::vector<Ct>& leaves) {
      const std::size_t ncols = (b.cluster == m.clusters) ? m.d : D;
      Ct sum;
      for (std::size_t j = 0; j < ncols; ++j) {
        const Ct col = load_ct(in_dir + "/col_" + std::to_string(b.cluster) + "_" +
                               std::to_string(b.idx) + "_" + std::to_string(j) + ".bin");
        Ct prod = cc->EvalMultNoRelin(col, leaves[j]);
        if (j == 0) sum = prod;
        else cc->EvalAddInPlace(sum, prod);
      }
      Ct out = cc->Rescale(cc->Relinearize(sum));
      if (b.cluster != m.clusters) {  // data blocks: plaintext penalty at deleted slots
        std::vector<double> pen;
        for (std::size_t s = 0; s < b.n_ids; ++s) {
          if (!deleted.count(id_of(b.cluster, b.idx, s))) continue;
          if (pen.empty()) pen.assign(cc->GetEncodingParams()->GetBatchSize(), 0.0);
          pen[s] = penalty;
        }
        if (!pen.empty()) {
          Plaintext pen_pt = cc->MakeCKKSPackedPlaintext(pen, 1, 1);
          out = cc->EvalAdd(out, pen_pt);
        }
      }
      return out;
    };

    // ---- Queries ------------------------------------------------------------------------------
    std::ofstream index(out_dir + "/index.txt");
    for (const auto& qname : m.queries) {
      const Ct qct = load_ct(in_dir + "/query_" + qname + ".bin");
      core::cpu_ref::OpenFheOps cpu_ops(cc, akeys);
      const std::vector<Ct> leaves = core::expand_tree(qct, D, ring, cpu_ops);

      auto session = engine.open_session(m.collection, m.key_tag, envelope_of_query(qname), sum_v);

      std::vector<std::uint32_t> all;
      for (std::uint32_t c = 0; c < m.clusters; ++c) all.push_back(c);
      // Per-stage results, compared block by block with the CPU chain.
      const auto verify = [&](const std::string& stage,
                              const std::vector<ombra::search::BlockScores>& res,
                              std::size_t expected_blocks) {
        expect(qname + " " + stage + ": block count", res.size() == expected_blocks);
        for (std::size_t i = 0; i < res.size(); ++i) {
          const auto& r = res[i];
          const Manifest::Block& b = block_of_file.at(r.segment);
          std::vector<std::string> ids;
          for (std::size_t s = 0; s < b.n_ids; ++s) ids.push_back(id_of(b.cluster, b.idx, s));
          expect(qname + " " + stage + " block " + std::to_string(i) + ": ids in slot order",
                 r.ids == ids && r.cluster == b.cluster);
          check(qname + " " + stage + " block " + std::to_string(i) + " (cluster " +
                    std::to_string(b.cluster) + "." + std::to_string(b.idx) + ")",
                ct_from_envelope(r.envelope, m), cpu_chain(b, leaves));
          const std::string f = qname + "_" + stage + "_" + std::to_string(i);
          std::ofstream o(out_dir + "/" + f + ".bin", std::ios::binary);
          o.write(reinterpret_cast<const char*>(r.envelope.data()),
                  static_cast<std::streamsize>(r.envelope.size()));
          index << f << " " << b.cluster << " " << b.idx << " " << r.tombstone_penalty_applied << "\n";
        }
      };

      std::size_t n_center = 0, n_data = 0;
      for (const auto& b : m.blocks) (b.cluster == m.clusters ? n_center : n_data) += 1;
      const auto s1 = session->score_centers();
      verify("stage1", s1, n_center);
      const auto s2 = session->score_clusters(all);
      verify("stage2", s2, n_data);

      // A repeated round on the same session (selective filters, decision 0019): the same words.
      const auto again = session->score_clusters({1, 0});
      std::vector<const ombra::search::BlockScores*> first_of;
      for (const auto& r : s2)
        if (r.cluster == 0 || r.cluster == 1) first_of.push_back(&r);
      bool same = true;
      std::size_t k = 0;
      for (std::uint32_t want : {1u, 0u})
        for (const auto* p : first_of)
          if (p->cluster == want) same = same && k < again.size() && again[k++].envelope == p->envelope;
      expect(qname + " repeated round on the same session gives identical ciphertexts",
             same && k == again.size());
      expect(qname + " out-of-range cluster is refused",
             throws_as<ombra::search::InvalidQuery>(
                 [&] { session->score_clusters({static_cast<std::uint32_t>(m.clusters)}); }));
    }

    const auto cm = engine.cache_metrics();
    std::printf("cache metrics: hits=%llu misses=%llu evictions=%llu hit_rate=%.4f bytes_paged=%llu\n",
                static_cast<unsigned long long>(cm.hits), static_cast<unsigned long long>(cm.misses),
                static_cast<unsigned long long>(cm.evictions), cm.hit_rate(),
                static_cast<unsigned long long>(cm.bytes_paged));
    std::printf("cache collection_bytes=%zu gpu_budget_bytes=%zu block_bytes=%zu\n", collection_bytes,
                cfg.gpu_budget_bytes, block_bytes);
    expect("collection is larger than the GPU budget and blocks were evicted",
           collection_bytes > cfg.gpu_budget_bytes && cm.evictions > 0);

    if (g_failures != 0) {
      std::printf("search engine exact match with OpenFHE: FAIL (%d checks)\n", g_failures);
      return 1;
    }
    std::printf("search engine exact match with OpenFHE: PASS\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
