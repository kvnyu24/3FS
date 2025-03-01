#pragma once

#include <unordered_map>
#include <vector>

#include "client/storage/StorageClient.h"
#include "common/utils/Coroutine.h"
#include "mgmtd/service/MgmtRequest.h"
#include "monitor_collector/service/MLWorkloadAnalytics.h"

namespace hf3fs::mgmtd {

class StorageMgmt {
 public:
  StorageMgmt(std::shared_ptr<storage::StorageClient> storageClient);
  
  CoTask<MgmtResponse> handleAddTarget(const MgmtRequest& request);
  CoTask<MgmtResponse> handleOfflineTarget(const MgmtRequest& request);
  CoTask<MgmtResponse> handleRemoveTarget(const MgmtRequest& request);
  CoTask<MgmtResponse> handleStorageSpaceInfo(const MgmtRequest& request);
  CoTask<MgmtResponse> handleStorageStats(const MgmtRequest& request);
  CoTask<MgmtResponse> handleTierInfo(const MgmtRequest& request);
  CoTask<MgmtResponse> handleMoveTier(const MgmtRequest& request);
  
  // New method for ML Workload Analytics
  CoTask<MgmtResponse> handleMLWorkloadAnalytics(const MgmtRequest& request);
  
 private:
  std::shared_ptr<storage::StorageClient> storageClient_;
  std::shared_ptr<monitor_collector::MLWorkloadAnalytics> mlWorkloadAnalytics_;
};

} // namespace hf3fs::mgmtd 