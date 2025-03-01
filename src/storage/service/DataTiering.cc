#include "storage/service/DataTiering.h"

#include <folly/Format.h>
#include <folly/logging/xlog.h>
#include <sys/time.h>

#include "common/utils/Time.h"
#include "storage/service/Components.h"
#include "storage/store/Chunk.h"
#include "storage/store/StorageTargets.h"

namespace hf3fs::storage {

namespace {
// Helper function to get current timestamp in seconds
uint64_t getCurrentTimeSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Helper function to convert StorageTier to string for logging
std::string tierToString(StorageTier tier) {
  switch (tier) {
    case StorageTier::HOT:
      return "HOT";
    case StorageTier::WARM:
      return "WARM";
    case StorageTier::COLD:
      return "COLD";
    case StorageTier::ARCHIVE:
      return "ARCHIVE";
    default:
      return "UNKNOWN";
  }
}
}  // namespace

DataTiering::DataTiering(const Config& config, Components& components)
    : config_(config),
      components_(components),
      concurrentTieringOps_(config.max_concurrent_tiering_operations()) {
  // Initialize tiering policy from configuration
  policy_.hotToColdThreshold = config.hot_to_cold_threshold_seconds();
  policy_.coldToArchiveThreshold = config.cold_to_archive_threshold_seconds();
  policy_.minSizeForTiering = config.min_size_for_tiering_bytes();
  policy_.enableCompressionForColdTier = config.enable_compression_for_cold();
  policy_.enableCompressionForArchiveTier = config.enable_compression_for_archive();
  policy_.hotTierMinFrequency = config.hot_tier_min_frequency();

  // Set up configuration update callback
  onConfigUpdated_ = config_.addCallbackGuard([this]() {
    policy_.hotToColdThreshold = config_.hot_to_cold_threshold_seconds();
    policy_.coldToArchiveThreshold = config_.cold_to_archive_threshold_seconds();
    policy_.minSizeForTiering = config_.min_size_for_tiering_bytes();
    policy_.enableCompressionForColdTier = config_.enable_compression_for_cold();
    policy_.enableCompressionForArchiveTier = config_.enable_compression_for_archive();
    policy_.hotTierMinFrequency = config_.hot_tier_min_frequency();

    concurrentTieringOps_.changeUsableTokens(config_.max_concurrent_tiering_operations());

    XLOG(INFO) << "DataTiering configuration updated";
  });
}

DataTiering::~DataTiering() {
  if (tieringRunner_) {
    auto result = stopAndJoin();
    if (!result) {
      XLOG(WARN) << "Failed to stop data tiering service: " << result.error();
    }
  }
}

Result<Void> DataTiering::init() {
  if (!config_.enabled()) {
    XLOG(INFO) << "Data tiering is disabled, not initializing";
    return success();
  }

  XLOG(INFO) << "Initializing data tiering service";
  
  // Create and start background runner for periodic tiering analysis
  tieringRunner_ = std::make_unique<BackgroundRunner>(
      "DataTiering", 
      [this]() { runTieringAnalysis(); }, 
      std::chrono::milliseconds(config_.tiering_check_interval_ms()));
  
  tieringRunner_->start();
  
  return success();
}

Result<Void> DataTiering::stopAndJoin() {
  if (!tieringRunner_) {
    return success();
  }

  XLOG(INFO) << "Stopping data tiering service";
  
  // Stop the background runner
  tieringRunner_->stop();
  tieringRunner_->join();
  tieringRunner_.reset();
  
  return success();
}

void DataTiering::recordAccess(const ChunkId& chunkId, bool isWrite) {
  if (!config_.enabled()) {
    return;
  }

  auto now = getCurrentTimeSeconds();
  
  // Update the chunk stats atomically
  chunkStats_.upsert(
      chunkId,
      [now, isWrite](ChunkAccessStats& stats) {
        stats.lastAccessTime = now;
        if (isWrite) {
          stats.writeCount++;
        } else {
          stats.readCount++;
        }
        
        // Simple exponential decay for access frequency calculation
        constexpr float alpha = 0.1;  // Weight for new observations
        float currentActivity = isWrite ? 2.0 : 1.0;  // Writes are weighted more than reads
        stats.accessFrequency = alpha * currentActivity + (1 - alpha) * stats.accessFrequency;
      },
      [now, isWrite]() {
        ChunkAccessStats newStats;
        newStats.lastAccessTime = now;
        if (isWrite) {
          newStats.writeCount = 1;
          newStats.accessFrequency = 2.0;  // Initial write is weighted more
        } else {
          newStats.readCount = 1;
          newStats.accessFrequency = 1.0;
        }
        return newStats;
      });
}

StorageTier DataTiering::getTier(const ChunkId& chunkId) {
  auto it = chunkStats_.find(chunkId);
  if (it != chunkStats_.end()) {
    return it->second.currentTier;
  }
  // Default to HOT tier if not found
  return StorageTier::HOT;
}

StorageTier DataTiering::calculateAppropriatesTier(const ChunkAccessStats& stats) {
  auto now = getCurrentTimeSeconds();
  auto timeSinceLastAccess = now - stats.lastAccessTime;
  
  // Archives candidates are those not accessed for a very long time
  if (timeSinceLastAccess > policy_.coldToArchiveThreshold) {
    return StorageTier::ARCHIVE;
  }
  
  // Cold candidates are those not accessed recently
  if (timeSinceLastAccess > policy_.hotToColdThreshold) {
    return StorageTier::COLD;
  }
  
  // For frequently accessed chunks, keep them in hot tier regardless of last access time
  if (stats.accessFrequency >= policy_.hotTierMinFrequency) {
    return StorageTier::HOT;
  }
  
  // Default to current tier if no conditions are met
  return stats.currentTier;
}

void DataTiering::runTieringAnalysis() {
  if (!config_.enabled()) {
    return;
  }

  XLOG(INFO) << "Starting tiering analysis run";
  
  size_t checkedCount = 0;
  size_t movedCount = 0;
  
  // Capture a snapshot of the current stats to avoid locking during iteration
  auto statsCopy = chunkStats_.getShallowCopies();
  
  for (const auto& [chunkId, stats] : statsCopy) {
    checkedCount++;
    
    // Calculate appropriate tier
    auto targetTier = calculateAppropriatesTier(stats);
    
    // Skip if no tier change needed
    if (targetTier == stats.currentTier) {
      continue;
    }
    
    // Skip small chunks based on policy
    try {
      // Get chunk info to check size
      auto chunk = components_.store->getChunk(chunkId);
      if (!chunk) {
        XLOG(WARN) << "Failed to get chunk info for " << chunkId << " during tiering analysis";
        continue;
      }
      
      if (chunk->getSize() < policy_.minSizeForTiering) {
        // Skip small chunks
        continue;
      }
      
      // Initiate a move operation
      auto result = co_await moveToTier(chunkId, targetTier);
      if (result) {
        movedCount++;
        XLOG(INFO) << "Moved chunk " << chunkId << " from " << tierToString(stats.currentTier)
                   << " to " << tierToString(targetTier);
      }
    } catch (const std::exception& ex) {
      XLOG(WARN) << "Failed to process chunk " << chunkId << " during tiering analysis: " << ex.what();
    }
  }
  
  XLOG(INFO) << "Completed tiering analysis: checked " << checkedCount << " chunks, moved " << movedCount;
}

CoTryTask<bool> DataTiering::checkAndMoveTier(const ChunkId& chunkId) {
  if (!config_.enabled()) {
    co_return false;
  }

  auto it = chunkStats_.find(chunkId);
  if (it == chunkStats_.end()) {
    // No stats available for this chunk
    co_return false;
  }
  
  auto& stats = it->second;
  auto targetTier = calculateAppropriatesTier(stats);
  
  if (targetTier == stats.currentTier) {
    // No tier change needed
    co_return false;
  }
  
  // Initiate move operation
  co_return co_await moveToTier(chunkId, targetTier);
}

CoTryTask<bool> DataTiering::moveToTier(const ChunkId& chunkId, StorageTier targetTier) {
  if (!config_.enabled()) {
    co_return false;
  }

  // Acquire semaphore to limit concurrent operations
  auto scopedSem = co_await concurrentTieringOps_.co_withSemaphore();
  
  auto it = chunkStats_.find(chunkId);
  if (it == chunkStats_.end()) {
    XLOG(WARN) << "No stats available for chunk " << chunkId << " during moveToTier";
    co_return false;
  }
  
  auto currentTier = it->second.currentTier;
  if (currentTier == targetTier) {
    // Already in the target tier
    co_return true;
  }
  
  XLOG(INFO) << "Moving chunk " << chunkId << " from " << tierToString(currentTier)
             << " to " << tierToString(targetTier);
  
  // Determine if compression is needed
  bool needCompression = (targetTier == StorageTier::COLD && policy_.enableCompressionForColdTier) ||
                         (targetTier == StorageTier::ARCHIVE && policy_.enableCompressionForArchiveTier);
  
  // Perform the actual move operation
  auto result = co_await moveChunkBetweenTiers(chunkId, currentTier, targetTier, needCompression);
  
  if (result) {
    // Update the chunk stats with new tier
    chunkStats_.upsert(
        chunkId,
        [targetTier](ChunkAccessStats& stats) {
          stats.currentTier = targetTier;
        },
        [targetTier]() {
          ChunkAccessStats newStats;
          newStats.currentTier = targetTier;
          return newStats;
        });
  }
  
  co_return result;
}

CoTryTask<bool> DataTiering::moveChunkBetweenTiers(const ChunkId& chunkId,
                                                   StorageTier sourceTier,
                                                   StorageTier targetTier,
                                                   bool compress) {
  // Get chunk info
  auto chunk = components_.store->getChunk(chunkId);
  if (!chunk) {
    XLOG(ERROR) << "Failed to get chunk info for " << chunkId << " during tier movement";
    co_return false;
  }
  
  // For now, this is a placeholder implementation
  // In a real implementation, this would:
  // 1. Read the chunk data from the current targets
  // 2. Apply compression if needed
  // 3. Write to the appropriate target for the new tier
  // 4. Update metadata to reflect the new location and tier
  
  // Apply compression if needed
  if (compress) {
    auto compressionResult = co_await compressChunkIfNeeded(chunkId, targetTier);
    if (!compressionResult) {
      XLOG(WARN) << "Compression failed for chunk " << chunkId;
      // Continue anyway, we'll just move without compression
    }
  }
  
  // Log the operation for now since we don't have actual tiering implementation
  XLOG(INFO) << "Would move chunk " << chunkId << " from " << tierToString(sourceTier)
             << " to " << tierToString(targetTier) << " (compression: " << (compress ? "yes" : "no") << ")";
  
  // Simulate success for now
  co_return true;
}

CoTryTask<bool> DataTiering::compressChunkIfNeeded(const ChunkId& chunkId, StorageTier targetTier) {
  // Decide compression algorithm based on the target tier
  // This is a placeholder for the actual implementation
  
  std::string compressionAlgo;
  
  if (targetTier == StorageTier::COLD) {
    // Use a faster compression algorithm for cold tier
    compressionAlgo = "LZ4";
  } else if (targetTier == StorageTier::ARCHIVE) {
    // Use a higher compression ratio algorithm for archive tier
    compressionAlgo = "ZSTD";
  } else {
    // No compression for hot tiers
    co_return true;
  }
  
  XLOG(INFO) << "Would compress chunk " << chunkId << " using " << compressionAlgo
             << " for " << tierToString(targetTier) << " tier";
  
  // Simulate success for now
  co_return true;
}

} // namespace hf3fs::storage 