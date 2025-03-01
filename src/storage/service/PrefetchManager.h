#pragma once

#include <chrono>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
class StorageOperator;

/**
 * Access pattern types identified by the prefetcher
 */
enum class AccessPatternType {
  UNKNOWN,        // No clear pattern identified
  SEQUENTIAL,     // Sequential access (e.g., 1, 2, 3, 4)
  STRIDED,        // Strided access (e.g., 1, 3, 5, 7)
  TRANSPOSE,      // Transpose-like pattern (common in ML)
  RANDOM,         // Random access pattern (no predictable pattern)
  ML_BATCH,       // Machine learning batch access pattern
  ML_SHUFFLE      // Machine learning shuffle pattern
};

/**
 * Structure to track recent access history
 */
struct AccessHistory {
  std::deque<ChunkId> recentChunks;
  std::chrono::steady_clock::time_point lastAccessTime;
  AccessPatternType detectedPattern{AccessPatternType::UNKNOWN};
  int patternStride{0};
  uint64_t clientId{0};      // ID of the client making the access
  uint32_t frameworkType{0}; // Type of ML framework (0=unknown, 1=PyTorch, 2=TensorFlow, etc.)
  
  // Map of future predictions with confidence scores (0.0-1.0)
  std::unordered_map<ChunkId, float> predictions;
};

/**
 * ML framework identification structure
 */
struct MLFrameworkInfo {
  uint32_t frameworkType;
  std::string frameworkName;
  float confidence;
};

/**
 * Intelligent prefetching service that predicts and prefetches chunks based on access patterns
 */
class PrefetchManager {
public:
  class Config : public ConfigBase<Config> {
    CONFIG_HOT_UPDATED_ITEM(enabled, true);
    CONFIG_HOT_UPDATED_ITEM(history_length, 32);
    CONFIG_HOT_UPDATED_ITEM(prefetch_threshold, 0.6f);
    CONFIG_HOT_UPDATED_ITEM(max_prefetch_chunks, 16);
    CONFIG_HOT_UPDATED_ITEM(prefetch_check_interval_ms, 10);
    CONFIG_HOT_UPDATED_ITEM(prediction_ttl_ms, 500);
    CONFIG_HOT_UPDATED_ITEM(max_concurrent_prefetch, 32);
    CONFIG_HOT_UPDATED_ITEM(max_active_predictions, 1000);
    CONFIG_HOT_UPDATED_ITEM(client_history_ttl_ms, 60000);  // 1 minute
  };

  PrefetchManager(const Config& config, Components& components, StorageOperator& storageOperator);
  ~PrefetchManager();

  /**
   * Initialize the prefetch manager
   */
  Result<Void> init();
  
  /**
   * Stop the prefetch manager
   */
  Result<Void> stopAndJoin();
  
  /**
   * Record a chunk access to update the history and predictions
   * 
   * @param chunkId The accessed chunk ID
   * @param clientId ID of the client making the access
   * @param offset Offset within the chunk that was accessed
   * @param size Size of the data accessed
   */
  void recordAccess(const ChunkId& chunkId, uint64_t clientId, uint64_t offset, uint64_t size);
  
  /**
   * Get the predicted next chunks that will be accessed
   * 
   * @param clientId ID of the client for which to get predictions
   * @return Vector of chunk IDs predicted to be accessed next, in order of confidence
   */
  std::vector<ChunkId> getPredictedChunks(uint64_t clientId);
  
  /**
   * Manually trigger prefetching for a client
   * 
   * @param clientId ID of the client for which to trigger prefetching
   * @return Number of chunks prefetched
   */
  CoTryTask<uint32_t> triggerPrefetch(uint64_t clientId);
  
  /**
   * Detect the ML framework being used based on access patterns
   * 
   * @param clientId ID of the client to analyze
   * @return Information about the detected ML framework
   */
  MLFrameworkInfo detectMLFramework(uint64_t clientId);

private:
  /**
   * Run the prefetch analysis and triggering logic
   */
  void runPrefetchAnalysis();
  
  /**
   * Detect access patterns from the history
   * 
   * @param history Access history to analyze
   * @return The detected access pattern type
   */
  AccessPatternType detectAccessPattern(const AccessHistory& history);
  
  /**
   * Generate predictions based on the detected access pattern
   * 
   * @param history Access history containing the detected pattern
   * @return Map of predicted chunk IDs to confidence scores
   */
  std::unordered_map<ChunkId, float> generatePredictions(const AccessHistory& history);
  
  /**
   * Prefetch chunks for a specific client based on predictions
   * 
   * @param clientId ID of the client
   * @param predictions Map of chunk IDs to confidence scores
   * @return Number of chunks prefetched
   */
  CoTryTask<uint32_t> prefetchChunks(uint64_t clientId, 
                                    const std::unordered_map<ChunkId, float>& predictions);
  
  /**
   * Prefetch a single chunk
   * 
   * @param chunkId ID of the chunk to prefetch
   * @return True if prefetch was successful
   */
  CoTryTask<bool> prefetchChunk(const ChunkId& chunkId);
  
  /**
   * Clean up expired predictions and histories
   */
  void cleanupExpiredEntries();

  Config config_;
  Components& components_;
  StorageOperator& storageOperator_;
  
  // Maps client ID to their access history
  folly::ConcurrentHashMap<uint64_t, AccessHistory> clientHistory_;
  
  // Set of chunks currently being prefetched to avoid duplicates
  folly::ConcurrentHashMap<ChunkId, bool> activePrefetches_;
  
  // Background runner for prefetch analysis
  std::unique_ptr<BackgroundRunner> prefetchRunner_;
  
  // Semaphore to limit concurrent prefetch operations
  Semaphore concurrentPrefetchOps_;
  
  // Configuration change callback
  utils::CallbackGuard onConfigUpdated_;
};

} // namespace hf3fs::storage 