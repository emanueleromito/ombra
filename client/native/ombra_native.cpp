// ombra._native: the CPU (OpenFHE) work of the local-mode server (T3.7, decision 0022).
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
//  - Engine.score: sum_j leaf_j * column_j (EvalMultNoRelin), one relinearization, one rescale,
//    optional plaintext penalty for tombstoned slots; as the server does.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstddef>
#include <cstdint>
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
#include "key/key-ser.h"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "openfhe_expand_ops.hpp"
#include "scheme/ckksrns/ckksrns-ser.h"

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
    ParallelOps(CC cc, const std::map<uint32_t, EvalKey<DCRTPoly>>& keys)
        : base_(std::move(cc), keys) {}

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
    ombra::core::cpu_ref::OpenFheOps base_;
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
        ombra::core::require_security_floor(cc_);  // hard rule 1, on the server side too

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
        ParallelOps ops(cc_, keys_);
        auto out = std::make_shared<Leaves>();
        out->cts = ombra::core::expand_tree(q, n_leaves, ring_, ops);
        for (const Ct& l : out->cts)
            if (l->GetNoiseScaleDeg() != 1 || l->GetLevel() != 0)
                throw std::runtime_error("expansion leaf is not at NoiseScaleDeg 1 and Level 0");
        return out;
    }

    std::shared_ptr<Columns> load_columns(const std::vector<py::bytes>& payloads) {
        std::vector<std::string> raw;
        raw.reserve(payloads.size());
        for (const auto& p : payloads) raw.push_back(to_str(p));
        py::gil_scoped_release nogil;
        auto out = std::make_shared<Columns>();
        out->cts.reserve(raw.size());
        for (const std::string& r : raw) {
            Ct c = deserialize<Ct>(r, "a stored column");
            if (c->GetKeyTag() != tag_)
                throw KeyTagError("stored column was made under another key than the collection's");
            out->bytes += r.size();
            out->cts.push_back(std::move(c));
        }
        return out;
    }

    // sum over j < n of leaf_j * column_j, relinearized and rescaled; `penalty` (one value per
    // slot, or empty) is added in the clear. Returns the OpenFHE BINARY bytes of the result.
    py::bytes score(const Leaves& leaves, const Columns& cols, const std::vector<double>& penalty) {
        if (cols.size() == 0 || leaves.size() < cols.size())
            throw std::invalid_argument("score: columns and leaves do not match");
        std::string out;
        {
            py::gil_scoped_release nogil;
            const long n = static_cast<long>(cols.size());
            int threads = 1;
#ifdef _OPENMP
            threads = omp_get_max_threads();
#endif
            std::vector<Ct> partial(static_cast<std::size_t>(threads));
            std::exception_ptr err;
#pragma omp parallel num_threads(threads)
            {
                int me = 0;
#ifdef _OPENMP
                me = omp_get_thread_num();
#endif
                Ct acc;
                try {
#pragma omp for schedule(static)
                    for (long j = 0; j < n; ++j) {
                        Ct t = cc_->EvalMultNoRelin(leaves.cts[j], cols.cts[j]);
                        acc = acc ? cc_->EvalAdd(acc, t) : t;
                    }
                } catch (...) {
#pragma omp critical(ombra_err)
                    err = std::current_exception();
                }
                partial[me] = acc;
            }
            if (err) std::rethrow_exception(err);
            Ct acc;
            for (auto& p : partial)
                if (p) acc = acc ? cc_->EvalAdd(acc, p) : p;
            Ct res = cc_->Rescale(cc_->Relinearize(acc));
            if (!penalty.empty()) {
                Plaintext pt = cc_->MakeCKKSPackedPlaintext(penalty, 1, res->GetLevel());
                res = cc_->EvalAdd(res, pt);
            }
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
    py::class_<Engine>(m, "Engine")
        .def(py::init<const py::bytes&, const py::bytes&, const py::bytes&, const std::string&>())
        .def_property_readonly("ring_dim", &Engine::ring_dim)
        .def_property_readonly("batch_size", &Engine::batch_size)
        .def_property_readonly("key_tag", &Engine::key_tag)
        .def_property_readonly("last_result_level", &Engine::last_result_level)
        .def("expand", &Engine::expand)
        .def("load_columns", &Engine::load_columns)
        .def("score", &Engine::score);
}
