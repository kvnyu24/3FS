#pragma once

#include <folly/concurrency/ConcurrentHashMap.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/Synchronized.h>
#include <string>
#include <unordered_map>

#include "analytics/StructuredTraceLog.h"
#include "client/mgmtd/IMgmtdClientForServer.h"
#include "client/mgmtd/RoutingInfo.h"
#include "client/storage/StorageMessenger.h"
#include "common/net/Server.h"
#include "common/net/Transport.h"
#include "common/net/ib/IBSocket.h"
#include "common/net/ib/RDMABuf.h"
#include "common/utils/Address.h"
#include "common/utils/ConfigBase.h"
#include "common/utils/Coroutine.h"
#include "common/utils/LockManager.h"
#include "common/utils/Semaphore.h"
#include "storage/aio/AioReadWorker.h"
#include "storage/service/BufferPool.h"
#include "storage/service/DataTiering.h"
#include "storage/service/ReliableForwarding.h"
#include "storage/service/ReliableUpdate.h"
#include "storage/store/StorageTargets.h"
#include "storage/update/UpdateWorker.h"
#include "common/Result.h"
#include "monitor_collector/service/MLWorkloadAnalytics.h"
#include "storage/service/AesCbcEncryption.h"
#include "storage/service/PrefetchManager.h"
#include "storage/service/StorageStore.h"
#include "storage/types/types_generated.h"

namespace hf3fs::storage {

struct Components;

class StorageOperator {
 public:
  class Config : public ConfigBase<Config> {
    CONFIG_OBJ(write_worker, UpdateWorker::Config);
    CONFIG_OBJ(event_trace_log, analytics::StructuredTraceLog<StorageEventTrace>::Config);
    CONFIG_OBJ(data_tiering, DataTiering::Config);
    CONFIG_HOT_UPDATED_ITEM(max_num_results_per_query, uint32_t{100});
    CONFIG_HOT_UPDATED_ITEM(batch_read_job_split_size, uint32_t{1024});
    CONFIG_HOT_UPDATED_ITEM(post_buffer_per_bytes, 64_KB);
    CONFIG_HOT_UPDATED_ITEM(batch_read_ignore_chain_version, false);
    CONFIG_HOT_UPDATED_ITEM(max_concurrent_rdma_writes, 256U);
    CONFIG_HOT_UPDATED_ITEM(max_concurrent_rdma_reads, 256U);
    CONFIG_HOT_UPDATED_ITEM(read_only, false);
    CONFIG_HOT_UPDATED_ITEM(rdma_transmission_req_timeout, 0_ms);
    CONFIG_HOT_UPDATED_ITEM(apply_transmission_before_getting_semaphore, true);
  };

  StorageOperator(const Config &config, Components &components, std::shared_ptr<StorageStore> storageStore)
      : config_(config),
        components_(components),
        updateWorker_(config_.write_worker()),
        dataTiering_(config_.data_tiering(), components),
        storageEventTrace_(config.event_trace_log()),
        storageStore_(storageStore) {
    for (const auto &ibdev : net::IBDevice::all()) {
      concurrentRdmaWriteSemaphore_.emplace(ibdev->id(), config.max_concurrent_rdma_writes());
      concurrentRdmaReadSemaphore_.emplace(ibdev->id(), config.max_concurrent_rdma_reads());
    }

    onConfigUpdated_ = config_.addCallbackGuard([this]() {
      for (auto &[_, semaphore] : concurrentRdmaWriteSemaphore_) {
        semaphore.changeUsableTokens(config_.max_concurrent_rdma_writes());
      }
      for (auto &[_, semaphore] : concurrentRdmaReadSemaphore_) {
        semaphore.changeUsableTokens(config_.max_concurrent_rdma_reads());
      }
    });
  }

  Result<Void> init(uint32_t numberOfDisks);

  Result<Void> stopAndJoin();

  CoTryTask<BatchReadRsp> batchRead(ServiceRequestContext &requestCtx,
                                    const BatchReadReq &req,
                                    serde::CallContext &ctx);

  CoTryTask<WriteRsp> write(ServiceRequestContext &requestCtx, const WriteReq &req, net::IBSocket *ibSocket);

  CoTryTask<UpdateRsp> update(ServiceRequestContext &requestCtx, const UpdateReq &req, net::IBSocket *ibSocket);

  CoTryTask<QueryLastChunkRsp> queryLastChunk(ServiceRequestContext &requestCtx, const QueryLastChunkReq &req);

  CoTryTask<TruncateChunksRsp> truncateChunks(ServiceRequestContext &requestCtx, const TruncateChunksReq &req);

  CoTryTask<RemoveChunksRsp> removeChunks(ServiceRequestContext &requestCtx, const RemoveChunksReq &req);

  CoTryTask<TargetSyncInfo> syncStart(const SyncStartReq &req);

  CoTryTask<SyncDoneRsp> syncDone(const SyncDoneReq &req);

  CoTryTask<SpaceInfoRsp> spaceInfo(const SpaceInfoReq &req);

  CoTryTask<CreateTargetRsp> createTarget(const CreateTargetReq &req);

  CoTryTask<OfflineTargetRsp> offlineTarget(const OfflineTargetReq &req);

  CoTryTask<RemoveTargetRsp> removeTarget(const RemoveTargetReq &req);

  CoTryTask<QueryChunkRsp> queryChunk(const QueryChunkReq &req);

  CoTryTask<GetAllChunkMetadataRsp> getAllChunkMetadata(const GetAllChunkMetadataReq &req);
  
  // Data tiering operations
  CoTryTask<MoveTierRsp> moveTier(const MoveTierReq &req);

  CoTryTask<GetTierInfoRsp> getTierInfo(const GetTierInfoReq &req);

  // ML Workload Analytics operations
  void setEncryptionEnabled(bool enabled);

 protected:
  using ChunkMetadataProcessor = std::function<CoTryTask<void>(const ChunkId &, const ChunkMetadata &)>;
 
  CoTask<IOResult> handleUpdate(ServiceRequestContext &requestCtx,
                               UpdateReq &req,
                               net::IBSocket *ibSocket,
                               TargetPtr &target);
 
  CoTask<IOResult> doUpdate(ServiceRequestContext &requestCtx,
                           const UpdateIO &updateIO,
                           const UpdateOptions &updateOptions,
                           uint32_t featureFlags,
                           const std::shared_ptr<StorageTarget> &target,
                           net::IBSocket *ibSocket,
                           BufferPool::Buffer &buffer,
                           net::RDMARemoteBuf &remoteBuf,
                           ChunkEngineUpdateJob &chunkEngineJob,
                           bool allowToAllocate);
 
  CoTask<IOResult> doCommit(ServiceRequestContext &requestCtx,
                           const CommitIO &commitIO,
                           const UpdateOptions &updateOptions,
                           ChunkEngineUpdateJob &chunkEngineJob,
                           uint32_t featureFlags,
                           const std::shared_ptr<StorageTarget> &target);
 
  Result<std::vector<std::pair<ChunkId, ChunkMetadata>>> doQuery(ServiceRequestContext &requestCtx,
                                                                const VersionedChainId &vChainId,
                                                                const ChunkIdRange &chunkIdRange);
 
  CoTryTask<uint32_t> processQueryResults(ServiceRequestContext &requestCtx,
                                         const VersionedChainId &vChainId,
                                         const ChunkIdRange &chunkIdRanges,
                                         ChunkMetadataProcessor processor,
                                         bool &moreChunksInRange);
 
  CoTask<IOResult> doTruncate(ServiceRequestContext &requestCtx,
                             const TruncateChunkOp &op,
                             flat::UserInfo userInfo,
                             uint32_t featureFlags);
 
  CoTask<IOResult> doRemove(ServiceRequestContext &requestCtx,
                           const RemoveChunksOp &op,
                           flat::UserInfo userInfo,
                           uint32_t featureFlags);

 private:
  friend class ReliableUpdate;

  ConstructLog<"storage::StorageOperator"> constructLog_;
  const Config &config_;
  Components &components_;
  UpdateWorker updateWorker_;
  DataTiering dataTiering_;
  analytics::StructuredTraceLog<StorageEventTrace> storageEventTrace_;
  std::unique_ptr<ConfigCallbackGuard> onConfigUpdated_;
  std::map<uint8_t, hf3fs::Semaphore> concurrentRdmaWriteSemaphore_;
  std::map<uint8_t, hf3fs::Semaphore> concurrentRdmaReadSemaphore_;
  std::atomic<uint64_t> totalReadBytes_{};
  std::atomic<uint64_t> totalReadIOs_{};
  std::shared_ptr<StorageStore> storageStore_;
  folly::Synchronized<std::unique_ptr<AesCbcEncryption>> encryption_;
  folly::Synchronized<std::unique_ptr<DataTiering>> dataTiering_;
  folly::Synchronized<std::unique_ptr<PrefetchManager>> prefetchManager_;
  folly::Synchronized<std::unique_ptr<monitor_collector::MLWorkloadAnalytics>> mlWorkloadAnalytics_;
  
  // Record an operation in the ML workload analytics
  void recordOperation(
      monitor_collector::OperationType opType,
      const std::string& path,
      uint64_t sizeBytes,
      uint64_t latencyUs,
      uint64_t offset,
      const std::string& clientInfo,
      const std::string& jobId);
      
  // Convert a client info string to a client ID
  uint64_t getClientIdFromInfo(const std::string& clientInfo);
};

}  // namespace hf3fs::storage
