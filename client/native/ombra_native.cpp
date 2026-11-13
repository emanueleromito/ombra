// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra._native: the CPU (OpenFHE) work of the local-mode server.
//
// Imported ONLY by the child process of local mode, never by the client process: the client
// process holds the secret key through the openfhe-python wheel, and one process must not hold two
// OpenFHE copies. The module sees ciphertexts, the crypto context and public evaluation keys as
// bytes (OpenFHE BINARY payloads, envelopes already opened by the caller). It never takes a secret
// key and prints nothing.
//
//  - Engine: deserializes the context and the EvalMult and automorphism keys of one collection
//    and checks the key tag against the keys it really deserialized.
//  - Engine.expand: the server's expansion tree (core/include/ombra/core/expansion_tree.hpp with
//    the CPU operations of core/src/openfhe_expand_ops.hpp), the same levels as the server: no
//    level is used, every leaf is checked to be at NoiseScaleDeg 1 and Level 0.
//  - Engine.load_columns also takes seeded columns (envelope kind 4, docs/decisions/0023): the
//    payload (c0', seed) is expanded to the ciphertext (c0', u) with the key tag checked.
//  - Seeder: the CLIENT side of seeded columns. Built from the context, public key and secret key
//    as bytes in memory (never written, never logged); encrypts a column with the public key
//    exactly as before, replaces c1 by a seeded uniform polynomial (seeded_ciphertext_openfhe.hpp),
//    and decrypts seeded columns. Used only inside spawned worker processes of the client
//    (ombra/_seeding.py), which never import openfhe-python.
//  - Engine.score: sum_j leaf_j * column_j (EvalMultNoRelin), one relinearization, one rescale,
//    optional plaintext penalty for tombstoned slots; as the server does.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/random.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "fast_cpu_ops.hpp"
#include "key/key-ser.h"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "ombra/core/seeded_ciphertext.hpp"
#include "openfhe.h"
#include "openfhe_expand_ops.hpp"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "seeded_ciphertext_openfhe.hpp"

namespace py = pybind11;
using namespace lbcrypto;
using Ct = Ciphertext<DCRTPoly>;
using CC = CryptoContext<DCRTPoly>;

namespace {

struct KeyTagError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::string to_str(const py::bytes& b) { return std::string(b); }

template <class T>
T deserialize(const std::string& bytes, const char* what) {
  std::istringstream in(bytes, std::ios::in | std::ios::binary);
  T obj;
  try {
    Serial::Deserialize(obj, in, SerType::BINARY);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("cannot read ") + what);
  }
  if (!obj) throw std::invalid_argument(std::string("cannot read ") + what);
  return obj;
}

struct Leaves {
  std::vector<Ct> cts;
  std::size_t size() const { return cts.size(); }
};

struct Columns {
  std::vector<Ct> cts;
  std::size_t bytes = 0;
  std::size_t size() const { return cts.size(); }
};

// CPU operations of the tree, one level at a time over several threads. EvalAutomorphism only
// reads the key map, so the nodes of a level are independent.
class ParallelOps {
 public:
  ParallelOps(CC cc, const std::map<uint32_t, EvalKey<DCRTPoly>>& keys,
              ombra::native::MonomialCache& cache)
      : base_(std::move(cc), keys, cache) {}

  std::pair<Ct, Ct> split(const Ct& c, std::size_t level, std::uint32_t galois, std::size_t shift,
                          bool want_odd) {
    return base_.split(c, level, galois, shift, want_odd);
  }

  std::vector<std::pair<Ct, Ct>> split_level(const std::vector<Ct>& nodes, std::size_t level,
                                             std::uint32_t galois, std::size_t shift,
                                             const std::vector<char>& want_odd) {
    std::vector<std::pair<Ct, Ct>> out(nodes.size());
    std::exception_ptr err;
    const long n = static_cast<long>(nodes.size());
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < n; ++i) {
      try {
        out[i] = base_.split(nodes[i], level, galois, shift, want_odd[i] != 0);
      } catch (...) {
#pragma omp critical(ombra_err)
        err = std::current_exception();
      }
    }
    if (err) std::rethrow_exception(err);
    return out;
  }

 private:
  ombra::native::FastSplitOps base_;
};

class Engine {
 public:
  // `tag` is the key tag the collection was created with. The keys must carry it.
  Engine(const py::bytes& context, const py::bytes& evalmult, const py::bytes& automorphism,
         const std::string& tag)
      : tag_(tag) {
    const std::string ctx_b = to_str(context), mult_b = to_str(evalmult),
                      auto_b = to_str(automorphism);
    py::gil_scoped_release nogil;
    {
      std::istringstream in(ctx_b, std::ios::in | std::ios::binary);
      try {
        Serial::Deserialize(cc_, in, SerType::BINARY);
      } catch (const std::exception&) {
        throw std::invalid_argument("cannot read the crypto context");
      }
    }
    if (!cc_) throw std::invalid_argument("cannot read the crypto context");
    ombra::core::require_security_floor(cc_);  // the 128-bit floor holds on the server side too

    using Impl = CryptoContextImpl<DCRTPoly>;
    // OpenFHE's key store is static and refuses a tag that already holds keys: a second
    // collection under the same key pair (re-clustering) uploads the same public keys again.
    std::lock_guard<std::mutex> g(store_mutex());
    Impl::ClearEvalMultKeys(tag);
    Impl::ClearEvalAutomorphismKeys(tag);
    std::set<std::string> mult_before, auto_before;
    for (const auto& kv : Impl::GetAllEvalMultKeys()) mult_before.insert(kv.first);
    for (const auto& kv : Impl::GetAllEvalAutomorphismKeys()) auto_before.insert(kv.first);
    const auto drop_foreign = [&] {
      std::vector<std::string> foreign;
      for (const auto& kv : Impl::GetAllEvalMultKeys())
        if (kv.first != tag && !mult_before.count(kv.first)) foreign.push_back(kv.first);
      for (const auto& t : foreign) Impl::ClearEvalMultKeys(t);
      foreign.clear();
      for (const auto& kv : Impl::GetAllEvalAutomorphismKeys())
        if (kv.first != tag && !auto_before.count(kv.first)) foreign.push_back(kv.first);
      for (const auto& t : foreign) Impl::ClearEvalAutomorphismKeys(t);
    };
    try {
      std::istringstream m(mult_b, std::ios::in | std::ios::binary);
      if (!Impl::DeserializeEvalMultKey(m, SerType::BINARY))
        throw std::invalid_argument("cannot read the EvalMult key");
      std::istringstream a(auto_b, std::ios::in | std::ios::binary);
      if (!Impl::DeserializeEvalAutomorphismKey(a, SerType::BINARY))
        throw std::invalid_argument("cannot read the automorphism keys");
    } catch (const OpenFHEException&) {
      drop_foreign();
      throw KeyTagError("evaluation keys were made under another key than the collection's");
    }
    try {
      const auto& v = Impl::GetEvalMultKeyVector(tag);
      if (v.empty()) throw std::runtime_error("empty");
      keys_ = Impl::GetEvalAutomorphismKeyMap(tag);
    } catch (const std::exception&) {
      drop_foreign();
      throw KeyTagError("evaluation keys were made under another key than the collection's");
    }
    drop_foreign();
    if (keys_.empty()) throw KeyTagError("no automorphism keys under the collection's key tag");
    ring_ = cc_->GetRingDimension();
    batch_ = cc_->GetEncodingParams()->GetBatchSize();
  }

  std::size_t ring_dim() const { return ring_; }
  std::size_t batch_size() const { return batch_; }
  const std::string& key_tag() const { return tag_; }

  // Query ciphertext (payload bytes) -> n_leaves leaves. Throws KeyTagError when the query was
  // made under another key, invalid_argument for a bad shape, level or scale.
  std::shared_ptr<Leaves> expand(const py::bytes& query, std::size_t n_leaves) {
    const std::string q_b = to_str(query);
    py::gil_scoped_release nogil;
    Ct q = deserialize<Ct>(q_b, "the query ciphertext");
    if (q->GetKeyTag() != tag_)
      throw KeyTagError("query ciphertext was made under another key than the collection's");
    if (q->GetElements().size() != 2 || q->GetLevel() != 0 || q->GetNoiseScaleDeg() != 1)
      throw std::invalid_argument("query ciphertext has an unexpected shape, level or scale");
    ParallelOps ops(cc_, keys_, mono_cache_);
    auto out = std::make_shared<Leaves>();
    out->cts = ombra::core::expand_tree(q, n_leaves, ring_, ops);
    for (const Ct& l : out->cts)
      if (l->GetNoiseScaleDeg() != 1 || l->GetLevel() != 0)
        throw std::runtime_error("expansion leaf is not at NoiseScaleDeg 1 and Level 0");
    return out;
  }

  // `kinds` (one per payload) is 2 for an OpenFHE BINARY ciphertext and 4 for a seeded payload;
  // empty means all 2. The key tag is checked for both kinds.
  std::shared_ptr<Columns> load_columns(const std::vector<py::bytes>& payloads,
                                        const std::vector<int>& kinds) {
    if (!kinds.empty() && kinds.size() != payloads.size())
      throw std::invalid_argument("load_columns: kinds and payloads differ in length");
    std::vector<std::string> raw;
    raw.reserve(payloads.size());
    for (const auto& p : payloads) raw.push_back(to_str(p));
    py::gil_scoped_release nogil;
    auto out = std::make_shared<Columns>();
    out->cts.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const std::string& r = raw[i];
      const int kind = kinds.empty() ? 2 : kinds[i];
      Ct c;
      if (kind == 2) {
        c = deserialize<Ct>(r, "a stored column");
        out->bytes += r.size();
      } else if (kind == 4) {
        c = expand_payload(r);
        out->bytes += 2 * r.size();  // the expanded pair is twice the stored half
      } else {
        throw std::invalid_argument("load_columns: unknown column kind");
      }
      if (c->GetKeyTag() != tag_)
        throw KeyTagError("stored column was made under another key than the collection's");
      out->cts.push_back(std::move(c));
    }
    return out;
  }

  // Seeded payload -> OpenFHE ciphertext. The payload's key tag is compared before expanding.
  Ct expand_payload(const std::string& r) const {
    ombra::core::SeededPayload p;
    try {
      p = ombra::core::read_seeded_payload(reinterpret_cast<const std::uint8_t*>(r.data()),
                                           r.size());
    } catch (const std::runtime_error& e) {
      throw std::invalid_argument(std::string("cannot read a seeded column: ") + e.what());
    }
    if (p.key_tag != tag_)
      throw KeyTagError("stored column was made under another key than the collection's");
    return ombra::native::expand_seeded(cc_, p);  // refuses other ring dimension or moduli
  }

  // sum over j < n of leaf_j * column_j, relinearized and rescaled; `penalty` (one value per
  // slot, or empty) is added in the clear. Returns the OpenFHE BINARY bytes of the result.
  py::bytes score(const Leaves& leaves, const Columns& cols, const std::vector<double>& penalty) {
    if (cols.size() == 0 || leaves.size() < cols.size())
      throw std::invalid_argument("score: columns and leaves do not match");
    std::string out;
    {
      py::gil_scoped_release nogil;
      Ct sum = ombra::native::score_sum(cc_, leaves.cts, cols.cts, cols.size());
      Ct res = ombra::native::score_finish(cc_, sum, penalty);
      result_level_ = res->GetLevel();
      std::ostringstream os(std::ios::out | std::ios::binary);
      Serial::Serialize(res, os, SerType::BINARY);
      out = os.str();
    }
    return py::bytes(out);
  }

  std::size_t last_result_level() const { return result_level_; }

 private:
  static std::mutex& store_mutex() {
    static std::mutex m;
    return m;
  }

  std::string tag_;
  CC cc_;
  std::map<uint32_t, EvalKey<DCRTPoly>> keys_;
  std::size_t ring_ = 0, batch_ = 0;
  std::size_t result_level_ = 0;
  ombra::native::MonomialCache mono_cache_;  // X^-2^l in evaluation form, per tree level
};

// 32 bytes from the operating system's CSPRNG: getrandom(2), looped over short reads and EINTR.
// Any other result is an error; the caller never gets a partly filled seed.
std::array<std::uint8_t, 32> fresh_seed_bytes() {
  std::array<std::uint8_t, 32> seed{};
  std::size_t got = 0;
  while (got < seed.size()) {
    const ssize_t n = getrandom(seed.data() + got, seed.size() - got, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("getrandom failed: " + std::string(std::strerror(errno)));
    }
    if (n == 0) throw std::runtime_error("getrandom returned no bytes");
    got += static_cast<std::size_t>(n);
  }
  return seed;
}

// Test hook: exactly 32 bytes from a file (a short read or an unreadable file raises).
std::array<std::uint8_t, 32> seed_from_file(const std::string& path) {
  std::array<std::uint8_t, 32> seed{};
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open the seed source");
  const std::size_t n = std::fread(seed.data(), 1, seed.size(), f);
  std::fclose(f);
  if (n != seed.size()) throw std::runtime_error("seed source returned too few bytes");
  return seed;
}

class Seeder {
 public:
  // Context, public key and secret key as OpenFHE BINARY bytes, held in memory only.
  Seeder(const py::bytes& context, const py::bytes& public_key, const py::bytes& secret_key) {
    const std::string cb = to_str(context), pb = to_str(public_key), sb = to_str(secret_key);
    py::gil_scoped_release nogil;
    {
      std::istringstream in(cb, std::ios::in | std::ios::binary);
      try {
        Serial::Deserialize(cc_, in, SerType::BINARY);
      } catch (const std::exception&) {
        throw std::invalid_argument("cannot read the crypto context");
      }
    }
    if (!cc_) throw std::invalid_argument("cannot read the crypto context");
    ombra::core::require_security_floor(cc_);
    pk_ = deserialize<PublicKey<DCRTPoly>>(pb, "the public key");
    sk_ = deserialize<PrivateKey<DCRTPoly>>(sb, "the secret key");
    if (pk_->GetKeyTag() != sk_->GetKeyTag())
      throw std::invalid_argument("public and secret key do not belong together");
    tag_ = pk_->GetKeyTag();
  }

  // MakeCKKSPackedPlaintext + Encrypt with the PUBLIC key (same overload and defaults as the
  // Python client), then the seeded replacement of c1. Returns the kind-4 payload.
  py::bytes encrypt_column(
      const py::array_t<double, py::array::c_style | py::array::forcecast>& v) {
    std::vector<double> values(v.data(), v.data() + v.size());
    std::vector<std::uint8_t> out;
    {
      py::gil_scoped_release nogil;
      Ct ct = cc_->Encrypt(pk_, cc_->MakeCKKSPackedPlaintext(values));
      out = ombra::native::make_seeded(cc_, sk_, ct, fresh_seed_bytes());
    }
    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
  }

  // BINARY serialization of the expanded ciphertext.
  py::bytes expand(const py::bytes& payload) {
    const std::string r = to_str(payload);
    std::string out;
    {
      py::gil_scoped_release nogil;
      Ct ct = expand_checked(r);
      std::ostringstream os(std::ios::out | std::ios::binary);
      Serial::Serialize(ct, os, SerType::BINARY);
      out = os.str();
    }
    return py::bytes(out);
  }

  // Decrypts a seeded column to `slots` values. Like ombra/_decrypt.py, all towers but the first
  // are dropped first (LevelReduce) so decoding is cheap; the safety condition is the one there.
  py::array_t<double> decrypt_column(const py::bytes& payload, std::size_t slots) {
    const std::string r = to_str(payload);
    std::vector<double> vals;
    {
      py::gil_scoped_release nogil;
      Ct ct = expand_checked(r);
      const std::size_t towers = ct->GetElements()[0].GetNumOfElements();
      if (towers > 1) ct = cc_->LevelReduce(ct, nullptr, static_cast<uint32_t>(towers - 1));
      Plaintext pt;
      cc_->Decrypt(sk_, ct, &pt);
      pt->SetLength(slots);
      vals = pt->GetRealPackedValue();
    }
    return py::array_t<double>(static_cast<py::ssize_t>(vals.size()), vals.data());
  }

  const std::string& key_tag() const { return tag_; }
  std::size_t slots() const { return cc_->GetEncodingParams()->GetBatchSize(); }

 private:
  Ct expand_checked(const std::string& r) const {
    ombra::core::SeededPayload p;
    try {
      p = ombra::core::read_seeded_payload(reinterpret_cast<const std::uint8_t*>(r.data()),
                                           r.size());
    } catch (const std::runtime_error& e) {
      throw std::invalid_argument(std::string("cannot read a seeded column: ") + e.what());
    }
    if (p.key_tag != tag_) throw KeyTagError("seeded column was made under another key");
    return ombra::native::expand_seeded(cc_, p);
  }

  CC cc_;
  PublicKey<DCRTPoly> pk_;
  PrivateKey<DCRTPoly> sk_;
  std::string tag_;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "OpenFHE CPU work of the Ombra local-mode server (never holds a secret key)";
  py::register_exception<KeyTagError>(m, "KeyTagError", PyExc_ValueError);
  m.def("openfhe_version", [] { return GetOPENFHEVersion(); });
  py::class_<Leaves, std::shared_ptr<Leaves>>(m, "Leaves")
      .def("__len__", &Leaves::size)
      .def_property_readonly("levels", [](const Leaves& l) {
        std::set<std::size_t> s;
        for (const Ct& c : l.cts) s.insert(c->GetLevel());
        return s;
      });
  py::class_<Columns, std::shared_ptr<Columns>>(m, "Columns")
      .def("__len__", &Columns::size)
      .def_readonly("nbytes", &Columns::bytes);
  m.def("fresh_seed", [] {
    const auto s = fresh_seed_bytes();
    return py::bytes(reinterpret_cast<const char*>(s.data()), s.size());
  });
  m.def("_seed_from_file", [](const std::string& path) {
    const auto s = seed_from_file(path);
    return py::bytes(reinterpret_cast<const char*>(s.data()), s.size());
  });
  m.attr("SEEDED_API") = 1;
  py::class_<Seeder>(m, "Seeder")
      .def(py::init<const py::bytes&, const py::bytes&, const py::bytes&>())
      .def("encrypt_column", &Seeder::encrypt_column)
      .def("expand", &Seeder::expand)
      .def("decrypt_column", &Seeder::decrypt_column)
      .def_property_readonly("key_tag", &Seeder::key_tag)
      .def_property_readonly("slots", &Seeder::slots);
  py::class_<Engine>(m, "Engine")
      .def(py::init<const py::bytes&, const py::bytes&, const py::bytes&, const std::string&>())
      .def_property_readonly("ring_dim", &Engine::ring_dim)
      .def_property_readonly("batch_size", &Engine::batch_size)
      .def_property_readonly("key_tag", &Engine::key_tag)
      .def_property_readonly("last_result_level", &Engine::last_result_level)
      .def("expand", &Engine::expand)
      .def("load_columns", &Engine::load_columns, py::arg("payloads"),
           py::arg("kinds") = std::vector<int>{})
      .def("score", &Engine::score);
}
