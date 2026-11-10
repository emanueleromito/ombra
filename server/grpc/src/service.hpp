// The ombra.v1.Ombra gRPC service over Store + SearchEngine (T2.7b).
//
// Security: plaintext localhost only here; TLS and API keys are T2.9. No message carries a secret
// key and the service never asks for one (hard rule 2). Logs carry counts, sizes and timings, never
// ids, ciphertext bytes or values (hard rule 4).
#pragma once

#include <mutex>
#include <set>
#include <string>

#include "ombra/search/session.hpp"
#include "ombra/storage/store.hpp"
#include "ombra/v1/ombra.grpc.pb.h"

namespace ombra::server {

class OmbraService final : public ombra::v1::Ombra::Service {
 public:
  OmbraService(storage::Store& store, search::SearchEngine& engine) : store_(store), engine_(engine) {}

  grpc::Status CreateCollection(grpc::ServerContext*, const v1::CreateCollectionRequest*,
                                v1::CreateCollectionResponse*) override;
  grpc::Status GetCollection(grpc::ServerContext*, const v1::GetCollectionRequest*,
                             v1::GetCollectionResponse*) override;
  grpc::Status ListCollections(grpc::ServerContext*, const v1::ListCollectionsRequest*,
                               v1::ListCollectionsResponse*) override;
  grpc::Status DropCollection(grpc::ServerContext*, const v1::DropCollectionRequest*,
                              v1::DropCollectionResponse*) override;
  grpc::Status UploadKeys(grpc::ServerContext*, grpc::ServerReader<v1::UploadKeysChunk>*,
                          v1::UploadKeysResponse*) override;
  grpc::Status Insert(grpc::ServerContext*, grpc::ServerReader<v1::InsertBlock>*,
                      v1::InsertResponse*) override;
  grpc::Status Delete(grpc::ServerContext*, const v1::DeleteRequest*, v1::DeleteResponse*) override;
  grpc::Status Search(grpc::ServerContext*,
                      grpc::ServerReaderWriter<v1::SearchResponse, v1::SearchRequest>*) override;
  grpc::Status ListBlocks(grpc::ServerContext*, const v1::ListBlocksRequest*,
                          v1::ListBlocksResponse*) override;
  grpc::Status GetBlock(grpc::ServerContext*, const v1::GetBlockRequest*,
                        grpc::ServerWriter<v1::GetBlockResponse>*) override;
  grpc::Status ReplaceBlock(grpc::ServerContext*, grpc::ServerReader<v1::ReplaceBlockRequest>*,
                            v1::ReplaceBlockResponse*) override;

 private:
  bool keys_present(const std::string& collection) const;
  // Loads the collection's keys into the engine unless already loaded (also after a restart).
  void ensure_loaded(const std::string& collection);
  void forget(const std::string& collection);

  storage::Store& store_;
  search::SearchEngine& engine_;
  std::mutex load_mu_;
  std::set<std::string> loaded_;  // collections whose keys the engine holds, guarded by load_mu_
};

}  // namespace ombra::server
