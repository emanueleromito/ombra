// The ombra.v1.Ombra gRPC service over Store + SearchEngine.
//
// Security: API keys with scopes are checked on every RPC; TLS is set up in main. No message
// carries a secret key and the service never asks for one. Logs carry counts, sizes and timings,
// never ids, ciphertext bytes or values.
#pragma once

#include <mutex>
#include <set>
#include <string>

#include "ombra/search/session.hpp"
#include "ombra/server/auth.hpp"
#include "ombra/server/metrics.hpp"
#include "ombra/storage/store.hpp"
#include "ombra/v1/ombra.grpc.pb.h"

namespace ombra::server {

class OmbraService final : public ombra::v1::Ombra::Service {
 public:
  // keys == nullptr: authentication is off (--no-auth). Otherwise every RPC needs a valid key whose
  // scope allows it (auth.hpp); refused calls end with UNAUTHENTICATED or PERMISSION_DENIED.
  OmbraService(storage::Store& store, search::SearchEngine& engine, const ApiKeyStore* keys,
               Metrics& metrics)
      : store_(store), engine_(engine), keys_(keys), metrics_(metrics) {}

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
  grpc::Status PutBlob(grpc::ServerContext*, grpc::ServerReader<v1::PutBlobRequest>*,
                       v1::PutBlobResponse*) override;
  grpc::Status GetBlob(grpc::ServerContext*, const v1::GetBlobRequest*,
                       grpc::ServerWriter<v1::GetBlobResponse>*) override;
  grpc::Status DeleteBlobs(grpc::ServerContext*, const v1::DeleteBlobsRequest*,
                           v1::DeleteBlobsResponse*) override;

 private:
  // The RPC bodies; the public overrides check the API key and record metrics around them.
  grpc::Status run_CreateCollection(grpc::ServerContext*,
      const v1::CreateCollectionRequest*,
      v1::CreateCollectionResponse*);
  grpc::Status run_GetCollection(grpc::ServerContext*,
      const v1::GetCollectionRequest*,
      v1::GetCollectionResponse*);
  grpc::Status run_ListCollections(grpc::ServerContext*,
      const v1::ListCollectionsRequest*,
      v1::ListCollectionsResponse*);
  grpc::Status run_DropCollection(grpc::ServerContext*,
      const v1::DropCollectionRequest*,
      v1::DropCollectionResponse*);
  grpc::Status run_UploadKeys(grpc::ServerContext*,
      grpc::ServerReader<v1::UploadKeysChunk>*,
      v1::UploadKeysResponse*);
  grpc::Status run_Insert(grpc::ServerContext*,
      grpc::ServerReader<v1::InsertBlock>*,
      v1::InsertResponse*);
  grpc::Status run_Delete(grpc::ServerContext*,
      const v1::DeleteRequest*,
      v1::DeleteResponse*);
  grpc::Status run_Search(grpc::ServerContext*,
      grpc::ServerReaderWriter<v1::SearchResponse, v1::SearchRequest>*);
  grpc::Status run_ListBlocks(grpc::ServerContext*,
      const v1::ListBlocksRequest*,
      v1::ListBlocksResponse*);
  grpc::Status run_GetBlock(grpc::ServerContext*,
      const v1::GetBlockRequest*,
      grpc::ServerWriter<v1::GetBlockResponse>*);
  grpc::Status run_ReplaceBlock(grpc::ServerContext*,
      grpc::ServerReader<v1::ReplaceBlockRequest>*,
      v1::ReplaceBlockResponse*);
  grpc::Status run_PutBlob(grpc::ServerContext*, grpc::ServerReader<v1::PutBlobRequest>*,
                           v1::PutBlobResponse*);
  grpc::Status run_GetBlob(grpc::ServerContext*, const v1::GetBlobRequest*,
                           grpc::ServerWriter<v1::GetBlobResponse>*);
  grpc::Status run_DeleteBlobs(grpc::ServerContext*, const v1::DeleteBlobsRequest*,
                               v1::DeleteBlobsResponse*);
  template <class Resp, class Fn>
  grpc::Status guarded(grpc::ServerContext* ctx, Rpc rpc, Resp* resp, Fn&& fn);

  bool keys_present(const std::string& collection) const;
  // Loads the collection's keys into the engine unless already loaded (also after a restart).
  void ensure_loaded(const std::string& collection);
  void forget(const std::string& collection);

  storage::Store& store_;
  search::SearchEngine& engine_;
  const ApiKeyStore* keys_;
  Metrics& metrics_;
  std::mutex load_mu_;
  std::set<std::string> loaded_;  // collections whose keys the engine holds, guarded by load_mu_
};

}  // namespace ombra::server
