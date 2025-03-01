#include "storage/service/PrefetchManager.h"

#include <algorithm>
#include <folly/Format.h>
#include <folly/logging/xlog.h>
#include <numeric>

#include "common/utils/Time.h"
#include "storage/service/Components.h"
#include "storage/service/StorageOperator.h"
#include "storage/store/Chunk.h"

namespace hf3fs::storage {

namespace {
// Helper function to get the current timestamp as a time_point
std::chrono::steady_clock::time_point getCurrentTimePoint() {
  return std::chrono::steady_clock::now();
}

// Helper function to convert AccessPatternType to string for logging
std::string patternTypeToString(AccessPatternType type) {
  switch (type) {
    case AccessPatternType::UNKNOWN:
      return "UNKNOWN";
    case AccessPatternType::SEQUENTIAL:
      return "SEQUENTIAL";
    case AccessPatternType::STRIDED:
      return "STRIDED";
    case AccessPatternType::TRANSPOSE:
      return "TRANSPOSE";
    case AccessPatternType::RANDOM:
      return "RANDOM";
    case AccessPatternType::ML_BATCH:
      return "ML_BATCH";
    case AccessPatternType::ML_SHUFFLE:
      return "ML_SHUFFLE";
    default:
      return "UNKNOWN";
  }
}

// Helper function to convert framework type to string
std::string frameworkTypeToString(uint32_t type) {
  switch (type) {
    case 1:
      return "PyTorch";
    case 2:
      return "TensorFlow";
    case 3:
      return "JAX";
    case 4:
      return "MXNet";
    default:
      return "Unknown";
  }
}
}  // namespace

PrefetchManager::PrefetchManager(const Config& config, 
                                Components& components,
                                StorageOperator& storageOperator)
    : config_(config),
      components_(components),
      storageOperator_(storageOperator),
      concurrentPrefetchOps_(config.max_concurrent_prefetch()) {
  
  // Set up configuration update callback
  onConfigUpdated_ = config_.addCallbackGuard([this]() {
    concurrentPrefetchOps_.changeUsableTokens(config_.max_concurrent_prefetch());
    XLOG(INFO) << "PrefetchManager configuration updated";
  });
}

PrefetchManager::~PrefetchManager() {
  if (prefetchRunner_) {
    auto result = stopAndJoin();
    if (!result) {
      XLOG(WARN) << "Failed to stop prefetch manager: " << result.error();
    }
  }
}

Result<Void> PrefetchManager::init() {
  if (!config_.enabled()) {
    XLOG(INFO) << "Prefetching is disabled, not initializing";
    return success();
  }

  XLOG(INFO) << "Initializing prefetch manager";
  
  // Create and start background runner for prefetch analysis
  prefetchRunner_ = std::make_unique<BackgroundRunner>(
      "PrefetchManager", 
      [this]() { runPrefetchAnalysis(); }, 
      std::chrono::milliseconds(config_.prefetch_check_interval_ms()));
  
  prefetchRunner_->start();
  
  return success();
}

Result<Void> PrefetchManager::stopAndJoin() {
  if (!prefetchRunner_) {
    return success();
  }

  XLOG(INFO) << "Stopping prefetch manager";
  
  prefetchRunner_->stop();
  prefetchRunner_->join();
  prefetchRunner_.reset();
  
  return success();
}

void PrefetchManager::recordAccess(const ChunkId& chunkId, 
                                  uint64_t clientId,
                                  uint64_t offset,
                                  uint64_t size) {
  if (!config_.enabled()) {
    return;
  }

  auto now = getCurrentTimePoint();
  
  // Update the client history
  clientHistory_.upsert(
      clientId,
      [&](AccessHistory& history) {
        // Update last access time
        history.lastAccessTime = now;
        history.clientId = clientId;
        
        // Add chunk to the recent access history
        history.recentChunks.push_back(chunkId);
        
        // Keep only the last N chunks in history
        while (history.recentChunks.size() > config_.history_length()) {
          history.recentChunks.pop_front();
        }
        
        // If history has enough entries, detect pattern and update predictions
        if (history.recentChunks.size() >= 3) {  // Need at least 3 entries to detect a pattern
          history.detectedPattern = detectAccessPattern(history);
          history.predictions = generatePredictions(history);
          
          XLOG(DBG8) << "Client " << clientId << " detected pattern: " 
                     << patternTypeToString(history.detectedPattern)
                     << ", generated " << history.predictions.size() << " predictions";
        }
      },
      [&]() {
        // Create new history entry for this client
        AccessHistory newHistory;
        newHistory.recentChunks.push_back(chunkId);
        newHistory.lastAccessTime = now;
        newHistory.clientId = clientId;
        return newHistory;
      });
}

std::vector<ChunkId> PrefetchManager::getPredictedChunks(uint64_t clientId) {
  std::vector<ChunkId> result;
  
  auto it = clientHistory_.find(clientId);
  if (it == clientHistory_.end()) {
    return result;  // No history for this client
  }
  
  // Get predictions and sort by confidence
  const auto& predictions = it->second.predictions;
  std::vector<std::pair<ChunkId, float>> sortedPredictions(predictions.begin(), predictions.end());
  
  // Sort by confidence (highest first)
  std::sort(sortedPredictions.begin(), sortedPredictions.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  
  // Extract sorted chunk IDs
  for (const auto& [chunkId, confidence] : sortedPredictions) {
    if (confidence >= config_.prefetch_threshold()) {
      result.push_back(chunkId);
      if (result.size() >= config_.max_prefetch_chunks()) {
        break;
      }
    }
  }
  
  return result;
}

CoTryTask<uint32_t> PrefetchManager::triggerPrefetch(uint64_t clientId) {
  if (!config_.enabled()) {
    co_return 0;
  }

  // Get the history for this client
  auto it = clientHistory_.find(clientId);
  if (it == clientHistory_.end()) {
    co_return 0;  // No history for this client
  }
  
  // Get the predictions
  const auto& predictions = it->second.predictions;
  if (predictions.empty()) {
    co_return 0;  // No predictions available
  }
  
  // Trigger prefetching based on predictions
  uint32_t prefetchCount = co_await prefetchChunks(clientId, predictions);
  
  XLOG(INFO) << "Triggered prefetch for client " << clientId 
             << ", prefetched " << prefetchCount << " chunks";
  
  co_return prefetchCount;
}

MLFrameworkInfo PrefetchManager::detectMLFramework(uint64_t clientId) {
  MLFrameworkInfo result{0, "Unknown", 0.0f};
  
  auto it = clientHistory_.find(clientId);
  if (it == clientHistory_.end()) {
    return result;  // No history for this client
  }
  
  const auto& history = it->second;
  
  // Simple heuristic detection based on access patterns
  if (history.detectedPattern == AccessPatternType::ML_BATCH) {
    // Likely PyTorch with batched access
    result.frameworkType = 1;
    result.frameworkName = "PyTorch";
    result.confidence = 0.8f;
  } else if (history.detectedPattern == AccessPatternType::STRIDED && 
             history.patternStride > 0 &&
             history.patternStride % 4 == 0) {
    // TensorFlow often uses power-of-2 strides
    result.frameworkType = 2;
    result.frameworkName = "TensorFlow";
    result.confidence = 0.7f;
  } else if (history.detectedPattern == AccessPatternType::ML_SHUFFLE) {
    // Could be JAX
    result.frameworkType = 3;
    result.frameworkName = "JAX";
    result.confidence = 0.6f;
  } else if (history.detectedPattern == AccessPatternType::TRANSPOSE) {
    // MXNet often performs transpose operations
    result.frameworkType = 4;
    result.frameworkName = "MXNet";
    result.confidence = 0.5f;
  }
  
  return result;
}

void PrefetchManager::runPrefetchAnalysis() {
  if (!config_.enabled()) {
    return;
  }

  // Clean up expired entries
  cleanupExpiredEntries();
  
  // Check active clients and trigger prefetching for those with predictions
  for (auto& [clientId, history] : clientHistory_.getShallowCopies()) {
    // Skip clients without predictions
    if (history.predictions.empty()) {
      continue;
    }
    
    // Trigger prefetching for this client
    try {
      auto prefetchCount = taskSync(prefetchChunks(clientId, history.predictions));
      if (prefetchCount && *prefetchCount > 0) {
        XLOG(DBG8) << "Auto-prefetched " << *prefetchCount << " chunks for client " << clientId;
      }
    } catch (const std::exception& ex) {
      XLOG(WARN) << "Failed to prefetch for client " << clientId << ": " << ex.what();
    }
  }
}

AccessPatternType PrefetchManager::detectAccessPattern(const AccessHistory& history) {
  const auto& chunks = history.recentChunks;
  
  if (chunks.size() < 3) {
    return AccessPatternType::UNKNOWN;
  }
  
  // Check for sequential pattern
  bool isSequential = true;
  for (size_t i = 1; i < chunks.size(); ++i) {
    if (chunks[i].value() != chunks[i-1].value() + 1) {
      isSequential = false;
      break;
    }
  }
  
  if (isSequential) {
    return AccessPatternType::SEQUENTIAL;
  }
  
  // Check for strided pattern
  int stride = 0;
  bool isStrided = true;
  
  // Try to detect the stride from the first few elements
  if (chunks.size() >= 3) {
    stride = chunks[1].value() - chunks[0].value();
    
    for (size_t i = 2; i < chunks.size(); ++i) {
      if (chunks[i].value() - chunks[i-1].value() != stride) {
        isStrided = false;
        break;
      }
    }
  }
  
  if (isStrided && stride != 0) {
    // Store the stride in the history object for future use
    const_cast<AccessHistory&>(history).patternStride = stride;
    return AccessPatternType::STRIDED;
  }
  
  // Check for transpose-like pattern (common in ML operations)
  bool isTranspose = false;
  // A transpose pattern typically shows as two interleaved sequences
  // For example: [1, 101, 2, 102, 3, 103, ...]
  if (chunks.size() >= 6) {  // Need enough elements to detect the pattern
    int stride1 = chunks[2].value() - chunks[0].value();
    int stride2 = chunks[3].value() - chunks[1].value();
    
    if (stride1 > 0 && stride2 > 0) {
      bool matchesTranspose = true;
      for (size_t i = 4; i < chunks.size(); i += 2) {
        if (i+1 < chunks.size() &&
            (chunks[i].value() != chunks[i-2].value() + stride1 ||
             chunks[i+1].value() != chunks[i-1].value() + stride2)) {
          matchesTranspose = false;
          break;
        }
      }
      
      if (matchesTranspose) {
        isTranspose = true;
      }
    }
  }
  
  if (isTranspose) {
    return AccessPatternType::TRANSPOSE;
  }
  
  // ML batch pattern: chunks with nearby IDs accessed in groups
  bool isMlBatch = false;
  if (chunks.size() >= 8) {  // Need enough elements to detect batch pattern
    // Group the chunks into potential batches
    std::vector<std::vector<ChunkId>> batches;
    std::vector<ChunkId> currentBatch;
    currentBatch.push_back(chunks[0]);
    
    for (size_t i = 1; i < chunks.size(); ++i) {
      // If this chunk is "close" to the previous one, add to current batch
      if (std::abs(static_cast<int64_t>(chunks[i].value()) - 
                  static_cast<int64_t>(currentBatch.back().value())) <= 10) {
        currentBatch.push_back(chunks[i]);
      } else {
        // Otherwise, start a new batch
        if (!currentBatch.empty()) {
          batches.push_back(currentBatch);
          currentBatch.clear();
        }
        currentBatch.push_back(chunks[i]);
      }
    }
    
    if (!currentBatch.empty()) {
      batches.push_back(currentBatch);
    }
    
    // If we have multiple batches with similar sizes, it's likely an ML batch pattern
    if (batches.size() >= 2) {
      size_t firstBatchSize = batches[0].size();
      bool similarSizes = true;
      
      for (size_t i = 1; i < batches.size(); ++i) {
        // Allow some variation in batch sizes
        if (std::abs(static_cast<int>(batches[i].size()) - static_cast<int>(firstBatchSize)) > 2) {
          similarSizes = false;
          break;
        }
      }
      
      if (similarSizes) {
        isMlBatch = true;
      }
    }
  }
  
  if (isMlBatch) {
    return AccessPatternType::ML_BATCH;
  }
  
  // ML shuffle pattern: pseudorandom access with some repeating elements
  bool isShuffle = false;
  if (chunks.size() >= 16) {  // Need many elements to detect shuffle
    // Count unique chunks
    std::unordered_set<uint64_t> uniqueChunks;
    for (const auto& chunk : chunks) {
      uniqueChunks.insert(chunk.value());
    }
    
    // If we have repeating elements but not completely random,
    // it might be a shuffle pattern
    float uniqueRatio = static_cast<float>(uniqueChunks.size()) / chunks.size();
    if (uniqueRatio > 0.3 && uniqueRatio < 0.7) {
      isShuffle = true;
    }
  }
  
  if (isShuffle) {
    return AccessPatternType::ML_SHUFFLE;
  }
  
  // If we can't identify a specific pattern, it's considered random
  return AccessPatternType::RANDOM;
}

std::unordered_map<ChunkId, float> PrefetchManager::generatePredictions(const AccessHistory& history) {
  std::unordered_map<ChunkId, float> predictions;
  
  // Generate predictions based on the detected pattern
  switch (history.detectedPattern) {
    case AccessPatternType::SEQUENTIAL: {
      // For sequential access, predict the next N chunks
      if (!history.recentChunks.empty()) {
        ChunkId lastChunk = history.recentChunks.back();
        for (uint32_t i = 1; i <= config_.max_prefetch_chunks(); ++i) {
          ChunkId nextChunk(lastChunk.value() + i);
          predictions[nextChunk] = 1.0f - (0.05f * i);  // Confidence decreases with distance
        }
      }
      break;
    }
    
    case AccessPatternType::STRIDED: {
      // For strided access, predict the next N chunks with the detected stride
      if (!history.recentChunks.empty() && history.patternStride != 0) {
        ChunkId lastChunk = history.recentChunks.back();
        for (uint32_t i = 1; i <= config_.max_prefetch_chunks(); ++i) {
          ChunkId nextChunk(lastChunk.value() + history.patternStride * i);
          predictions[nextChunk] = 1.0f - (0.05f * i);
        }
      }
      break;
    }
    
    case AccessPatternType::TRANSPOSE: {
      // For transpose-like pattern, predict alternating chunks
      if (history.recentChunks.size() >= 2) {
        ChunkId lastChunk1 = history.recentChunks[history.recentChunks.size() - 1];
        ChunkId lastChunk2 = history.recentChunks[history.recentChunks.size() - 2];
        
        // Find the two strides
        int stride1 = 0, stride2 = 0;
        if (history.recentChunks.size() >= 4) {
          stride1 = lastChunk1.value() - history.recentChunks[history.recentChunks.size() - 3].value();
          stride2 = lastChunk2.value() - history.recentChunks[history.recentChunks.size() - 4].value();
        }
        
        // If we couldn't determine strides, use reasonable defaults
        if (stride1 == 0) stride1 = 1;
        if (stride2 == 0) stride2 = 1;
        
        // Predict alternating chunks
        for (uint32_t i = 1; i <= config_.max_prefetch_chunks() / 2; ++i) {
          ChunkId nextChunk1(lastChunk1.value() + stride1);
          ChunkId nextChunk2(lastChunk2.value() + stride2);
          predictions[nextChunk1] = 0.9f - (0.05f * i);
          predictions[nextChunk2] = 0.85f - (0.05f * i);
          lastChunk1 = nextChunk1;
          lastChunk2 = nextChunk2;
        }
      }
      break;
    }
    
    case AccessPatternType::ML_BATCH: {
      // For ML batch pattern, predict the next batch
      if (history.recentChunks.size() >= 4) {
        // Try to detect batch size and boundaries
        std::vector<std::vector<ChunkId>> batches;
        std::vector<ChunkId> currentBatch;
        currentBatch.push_back(history.recentChunks[0]);
        
        for (size_t i = 1; i < history.recentChunks.size(); ++i) {
          if (std::abs(static_cast<int64_t>(history.recentChunks[i].value()) - 
                      static_cast<int64_t>(currentBatch.back().value())) <= 10) {
            currentBatch.push_back(history.recentChunks[i]);
          } else {
            if (!currentBatch.empty()) {
              batches.push_back(currentBatch);
              currentBatch.clear();
            }
            currentBatch.push_back(history.recentChunks[i]);
          }
        }
        
        if (!currentBatch.empty()) {
          batches.push_back(currentBatch);
        }
        
        // If we detected at least two batches, try to predict the next one
        if (batches.size() >= 2) {
          auto& lastBatch = batches.back();
          auto& secondLastBatch = batches[batches.size() - 2];
          
          // Calculate offsets between batches
          int64_t batchOffset = 0;
          if (!lastBatch.empty() && !secondLastBatch.empty()) {
            batchOffset = lastBatch[0].value() - secondLastBatch[0].value();
          }
          
          // If we found a reasonable offset, predict the next batch
          if (batchOffset > 0) {
            for (size_t i = 0; i < lastBatch.size() && predictions.size() < config_.max_prefetch_chunks(); ++i) {
              ChunkId nextChunk(lastBatch[i].value() + batchOffset);
              predictions[nextChunk] = 0.85f - (0.01f * i);
            }
          }
        }
      }
      break;
    }
    
    case AccessPatternType::ML_SHUFFLE: {
      // For shuffle pattern, predict based on previously seen chunks
      if (history.recentChunks.size() >= 8) {
        // Count frequencies of chunks
        std::unordered_map<uint64_t, int> chunkCounts;
        for (const auto& chunk : history.recentChunks) {
          chunkCounts[chunk.value()]++;
        }
        
        // Find chunks with highest frequency but not recently accessed
        std::vector<std::pair<uint64_t, int>> sortedCounts(chunkCounts.begin(), chunkCounts.end());
        std::sort(sortedCounts.begin(), sortedCounts.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Get the most recent chunks to avoid re-predicting them
        std::unordered_set<uint64_t> recentSet;
        size_t recentCount = std::min(history.recentChunks.size(), size_t(4));
        for (size_t i = history.recentChunks.size() - recentCount; i < history.recentChunks.size(); ++i) {
          recentSet.insert(history.recentChunks[i].value());
        }
        
        // Add predictions for frequent chunks not recently accessed
        for (const auto& [chunkValue, count] : sortedCounts) {
          if (predictions.size() >= config_.max_prefetch_chunks()) {
            break;
          }
          
          if (recentSet.find(chunkValue) == recentSet.end()) {
            ChunkId chunkId(chunkValue);
            float confidence = 0.6f + (0.05f * count);  // Higher count = higher confidence
            if (confidence > 0.9f) confidence = 0.9f;   // Cap at 0.9
            predictions[chunkId] = confidence;
          }
        }
      }
      break;
    }
    
    case AccessPatternType::RANDOM:
    case AccessPatternType::UNKNOWN:
    default:
      // For random or unknown patterns, make minimal predictions
      if (!history.recentChunks.empty()) {
        ChunkId lastChunk = history.recentChunks.back();
        // Just predict the next sequential chunk with low confidence
        predictions[ChunkId(lastChunk.value() + 1)] = 0.3f;
      }
      break;
  }
  
  return predictions;
}

CoTryTask<uint32_t> PrefetchManager::prefetchChunks(uint64_t clientId,
                                                   const std::unordered_map<ChunkId, float>& predictions) {
  if (predictions.empty()) {
    co_return 0;
  }

  // Sort predictions by confidence
  std::vector<std::pair<ChunkId, float>> sortedPredictions(predictions.begin(), predictions.end());
  std::sort(sortedPredictions.begin(), sortedPredictions.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  
  uint32_t prefetchCount = 0;
  
  // Prefetch chunks with confidence above the threshold
  for (const auto& [chunkId, confidence] : sortedPredictions) {
    if (confidence < config_.prefetch_threshold()) {
      continue;  // Skip low-confidence predictions
    }
    
    if (prefetchCount >= config_.max_prefetch_chunks()) {
      break;  // Limit the number of prefetched chunks
    }
    
    // Check if this chunk is already being prefetched
    bool alreadyPrefetching = false;
    activePrefetches_.upsert(
        chunkId,
        [&](bool&) { alreadyPrefetching = true; },
        [&]() {
          alreadyPrefetching = false;
          return true;
        });
    
    if (alreadyPrefetching) {
      continue;  // Skip if already prefetching
    }
    
    // Prefetch the chunk
    try {
      bool success = co_await prefetchChunk(chunkId);
      if (success) {
        prefetchCount++;
      }
    } catch (const std::exception& ex) {
      XLOG(WARN) << "Failed to prefetch chunk " << chunkId << ": " << ex.what();
    }
    
    // Remove from active prefetches
    activePrefetches_.erase(chunkId);
  }
  
  co_return prefetchCount;
}

CoTryTask<bool> PrefetchManager::prefetchChunk(const ChunkId& chunkId) {
  // Acquire semaphore to limit concurrent operations
  auto scopedSem = co_await concurrentPrefetchOps_.co_withSemaphore();
  
  // In a real implementation, this would:
  // 1. Check if the chunk exists
  // 2. Read it into cache
  // 3. Return success/failure
  
  // For now, just simulate a successful prefetch with a small delay
  XLOG(DBG8) << "Prefetching chunk " << chunkId;
  
  // Simulate a small delay for the prefetch operation
  co_await folly::coro::sleep(std::chrono::milliseconds(5));
  
  // In real implementation, we would call into the StorageOperator to perform the prefetch
  // auto result = co_await storageOperator_.prefetchChunk(chunkId);
  
  // For now, just assume success
  co_return true;
}

void PrefetchManager::cleanupExpiredEntries() {
  auto now = getCurrentTimePoint();
  
  // Clean up expired client histories
  auto clientHistoryCopy = clientHistory_.getShallowCopies();
  for (const auto& [clientId, history] : clientHistoryCopy) {
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - history.lastAccessTime).count();
    
    if (elapsedMs > config_.client_history_ttl_ms()) {
      XLOG(DBG8) << "Removing expired history for client " << clientId;
      clientHistory_.erase(clientId);
    }
  }
  
  // Ensure we don't exceed maximum active predictions
  if (activePrefetches_.size() > config_.max_active_predictions()) {
    XLOG(WARN) << "Too many active prefetches (" << activePrefetches_.size() 
               << "), clearing some";
    
    // In a real implementation, we might want to be more selective about which ones to clear
    activePrefetches_.clear();
  }
}

} // namespace hf3fs::storage 