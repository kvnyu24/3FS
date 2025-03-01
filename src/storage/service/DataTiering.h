#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <folly/concurrency/ConcurrentHashMap.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include "common/utils/ConfigBase.h"
#include "common/utils/Coroutine.h"
#include "common/utils/Result.h"
#include "common/utils/Semaphore.h"
#include "common/utils/BackgroundRunner.h"
#include "storage/store/StorageTargets.h"

namespace hf3fs::storage {

// Forward declarations
struct Components;
class BufferPool;

/**
 * Enum representing different storage tiers
 */
enum class StorageTier {
  HOT,     // High performance tier (e.g., NVMe SSDs)
  WARM,    // Medium performance tier (e.g., SATA SSDs)
  COLD,    // Low performance tier (e.g., HDDs or network storage)
  ARCHIVE  // Archive tier (e.g., cold storage)
};

/**
 * Access statistics tracked for each chunk to inform tiering decisions
 */
struct ChunkAccessStats {
  uint64_t lastAccessTime{0};
  uint64_t readCount{0};
  uint64_t writeCount{0};
  float accessFrequency{0.0};
  StorageTier currentTier{StorageTier::HOT};
};

/**
 * Data tiering policy configuration
 */
struct TieringPolicy {
  // Time thresholds in seconds
  uint64_t hotToColdThreshold{7 * 24 * 3600};  // 7 days without access
  uint64_t coldToArchiveThreshold{30 * 24 * 3600};  // 30 days without access
  
  // Frequency thresholds
  float hotTierMinFrequency{0.1};  // Min access frequency to stay in hot tier
  
  // Size thresholds to prevent moving very small files
  uint64_t minSizeForTiering{1024 * 1024};  // 1MB
  
  // Compression settings
  bool enableCompressionForColdTier{true};
  bool enableCompressionForArchiveTier{true};
};

/**
 * DataTiering service manages the lifecycle of data across different storage tiers
 */
class DataTiering {
public:
  class Config : public ConfigBase<Config> {
    CONFIG_HOT_UPDATED_ITEM(enabled, true);
    CONFIG_HOT_UPDATED_ITEM(tiering_check_interval_ms, 3600000);  // Default: 1 hour
    CONFIG_HOT_UPDATED_ITEM(hot_to_cold_threshold_seconds, 604800);  // Default: 7 days
    CONFIG_HOT_UPDATED_ITEM(cold_to_archive_threshold_seconds, 2592000);  // Default: 30 days
    CONFIG_HOT_UPDATED_ITEM(min_size_for_tiering_bytes, 1048576);  // Default: 1MB
    CONFIG_HOT_UPDATED_ITEM(enable_compression_for_cold, true);
    CONFIG_HOT_UPDATED_ITEM(enable_compression_for_archive, true);
    CONFIG_HOT_UPDATED_ITEM(hot_tier_min_frequency, 0.1);
    CONFIG_HOT_UPDATED_ITEM(max_concurrent_tiering_operations, 16);
  };

  DataTiering(const Config& config, Components& components);
  ~DataTiering();

  /**
   * Initialize the data tiering service
   */
  Result<Void> init();
  
  /**
   * Stop the data tiering service
   */
  Result<Void> stopAndJoin();
  
  /**
   * Record an access to a chunk, updating its statistics
   */
  void recordAccess(const ChunkId& chunkId, bool isWrite);
  
  /**
   * Move a chunk to the specified tier
   */
  CoTryTask<bool> moveToTier(const ChunkId& chunkId, StorageTier targetTier);
  
  /**
   * Get current tier for a chunk
   */
  StorageTier getTier(const ChunkId& chunkId);
  
  /**
   * Manually trigger tiering check for a chunk
   */
  CoTryTask<bool> checkAndMoveTier(const ChunkId& chunkId);

private:
  /**
   * Run tiering analysis on all chunks
   */
  void runTieringAnalysis();
  
  /**
   * Calculate the appropriate tier for a chunk based on its access pattern
   */
  StorageTier calculateAppropriatesTier(const ChunkAccessStats& stats);
  
  /**
   * Internal method to move data between tiers
   */
  CoTryTask<bool> moveChunkBetweenTiers(const ChunkId& chunkId, 
                                        StorageTier sourceTier, 
                                        StorageTier targetTier,
                                        bool compress);
                                        
  /**
   * Apply compression to a chunk if needed
   */
  CoTryTask<bool> compressChunkIfNeeded(const ChunkId& chunkId, StorageTier targetTier);

  Config config_;
  Components& components_;
  folly::ConcurrentHashMap<ChunkId, ChunkAccessStats> chunkStats_;
  TieringPolicy policy_;
  
  // Background runner for tiering analysis
  std::unique_ptr<BackgroundRunner> tieringRunner_;
  
  // Semaphore to limit concurrent tiering operations
  Semaphore concurrentTieringOps_;
  
  // Configuration change callback
  utils::CallbackGuard onConfigUpdated_;
};

} // namespace hf3fs::storage 