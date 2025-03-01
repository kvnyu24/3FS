#include "src/monitor_collector/service/MLWorkloadAnalytics.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <regex>

namespace folly {
    dynamic dynamic::object;
    dynamic dynamic::array;
}

namespace hf3fs::monitor_collector {

namespace {
// Helper function to get current timestamp in milliseconds
uint64_t getCurrentTimeMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Helper function to convert WorkloadType to string
std::string workloadTypeToString(WorkloadType type) {
  switch (type) {
    case WorkloadType::TRAINING:
      return "TRAINING";
    case WorkloadType::INFERENCE:
      return "INFERENCE";
    case WorkloadType::DATA_PREP:
      return "DATA_PREP";
    case WorkloadType::EVALUATION:
      return "EVALUATION";
    case WorkloadType::CHECKPOINT:
      return "CHECKPOINT";
    case WorkloadType::UNKNOWN:
    default:
      return "UNKNOWN";
  }
}

// Helper function to convert FrameworkType to string
std::string frameworkTypeToString(FrameworkType type) {
  switch (type) {
    case FrameworkType::PYTORCH:
      return "PYTORCH";
    case FrameworkType::TENSORFLOW:
      return "TENSORFLOW";
    case FrameworkType::JAX:
      return "JAX";
    case FrameworkType::MXNET:
      return "MXNET";
    case FrameworkType::CUSTOM:
      return "CUSTOM";
    case FrameworkType::UNKNOWN:
    default:
      return "UNKNOWN";
  }
}

// Helper function to convert OperationType to string
std::string operationTypeToString(OperationType type) {
  switch (type) {
    case OperationType::READ:
      return "READ";
    case OperationType::WRITE:
      return "WRITE";
    case OperationType::OPEN:
      return "OPEN";
    case OperationType::CLOSE:
      return "CLOSE";
    case OperationType::STAT:
      return "STAT";
    case OperationType::LIST:
      return "LIST";
    case OperationType::CREATE:
      return "CREATE";
    case OperationType::MKDIR:
      return "MKDIR";
    case OperationType::REMOVE:
      return "REMOVE";
    case OperationType::RENAME:
      return "RENAME";
    default:
      return "UNKNOWN";
  }
}

// Helper function to get file extension
std::string getFileExtension(const std::string& path) {
  std::filesystem::path filePath(path);
  return filePath.extension().string();
}

// Helper function to detect if a path looks like an ML dataset or model
bool isMLPath(const std::string& path) {
  // Look for common ML-related keywords in the path
  static const std::vector<std::string> mlKeywords = {
      "model", "dataset", "checkpoint", "weights", "train", "test", "val", 
      "data", "inference", "eval", "batch", "tensor", "epoch", "params"
  };
  
  std::string lowerPath = path;
  std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
  
  for (const auto& keyword : mlKeywords) {
    if (lowerPath.find(keyword) != std::string::npos) {
      return true;
    }
  }
  
  return false;
}
} // namespace

MLWorkloadAnalytics::MLWorkloadAnalytics(const Config& config)
    : config_(config) {
  // Register config update callback
  onConfigUpdated_ = config_.addCallbackGuard([]() {
    std::cout << "MLWorkloadAnalytics configuration updated" << std::endl;
  });
}

MLWorkloadAnalytics::~MLWorkloadAnalytics() {
  if (analysisRunner_ || reportRunner_) {
    auto result = stopAndJoin();
    if (!result) {
      std::cerr << "Failed to stop ML workload analytics: " << result.error() << std::endl;
    }
  }
}

Result<Void> MLWorkloadAnalytics::init() {
  if (!config_.enabled()) {
    std::cout << "ML workload analytics is disabled, not initializing" << std::endl;
    return success();
  }

  std::cout << "Initializing ML workload analytics" << std::endl;
  
  // Create and start background runner for workload analysis
  analysisRunner_ = std::make_unique<BackgroundRunner>(
      "MLWorkloadAnalyzer", 
      [this]() { runWorkloadAnalysis(); }, 
      std::chrono::milliseconds(config_.analysis_interval_ms()));
  
  analysisRunner_->start();
  
  // Create and start background runner for report generation
  reportRunner_ = std::make_unique<BackgroundRunner>(
      "MLWorkloadReporter", 
      [this]() { generateReport(); }, 
      std::chrono::milliseconds(config_.report_interval_ms()));
  
  reportRunner_->start();
  
  return success();
}

Result<Void> MLWorkloadAnalytics::stopAndJoin() {
  if (analysisRunner_) {
    std::cout << "Stopping workload analysis runner" << std::endl;
    analysisRunner_->stop();
    analysisRunner_->join();
    analysisRunner_.reset();
  }

  if (reportRunner_) {
    std::cout << "Stopping workload report runner" << std::endl;
    reportRunner_->stop();
    reportRunner_->join();
    reportRunner_.reset();
  }
  
  return success();
}

void MLWorkloadAnalytics::recordOperation(const OperationRecord& record) {
  if (!config_.enabled()) {
    return;
  }

  auto now = getCurrentTimeMs();
  uint64_t clientId = record.clientId;
  
  // Update the recent operations history
  recentOperations_.upsert(
      clientId,
      [&record](std::vector<OperationRecord>& operations) {
        operations.push_back(record);
        
        // Limit the size of recent operations
        constexpr size_t MAX_RECENT_OPS = 1000;
        if (operations.size() > MAX_RECENT_OPS) {
          operations.erase(operations.begin(), operations.begin() + (operations.size() - MAX_RECENT_OPS));
        }
      },
      [&record]() {
        std::vector<OperationRecord> newOperations;
        newOperations.push_back(record);
        return newOperations;
      });
  
  // Update the workload stats
  activeWorkloads_.upsert(
      clientId,
      [&record, now](std::shared_ptr<WorkloadStats>& stats) {
        if (!stats) {
          stats = std::make_shared<WorkloadStats>();
          stats->clientId = record.clientId;
          stats->startTimestamp = now;
          stats->jobId = record.jobId;
        }
        
        stats->lastActivityTimestamp = now;
        
        // Update operation counts and sizes
        if (record.type == OperationType::READ) {
          stats->readOps++;
          stats->totalBytesRead += record.sizeBytes;
          
          // Detect if this is a sequential or random read
          // This is a simplified heuristic - in reality would need more context
          if (!stats->mostAccessedFiles.empty() && 
              stats->mostAccessedFiles.back() == record.path && 
              record.offset > 0) {
            stats->sequentialReadOps++;
          } else {
            stats->randomReadOps++;
          }
        } else if (record.type == OperationType::WRITE) {
          stats->writeOps++;
          stats->totalBytesWritten += record.sizeBytes;
        }
        
        // Update latency tracking
        if (record.latencyUs > 0) {
          // Simple running average - could be improved
          stats->avgLatencyUs = (stats->avgLatencyUs * (stats->readOps + stats->writeOps - 1) + 
                                record.latencyUs) / (stats->readOps + stats->writeOps);
          
          if (record.latencyUs > stats->maxLatencyUs) {
            stats->maxLatencyUs = record.latencyUs;
          }
        }
        
        // Track file types
        std::string ext = getFileExtension(record.path);
        if (!ext.empty()) {
          stats->fileTypes[ext]++;
        }
        
        // Track most accessed files
        auto it = std::find(stats->mostAccessedFiles.begin(), stats->mostAccessedFiles.end(), record.path);
        if (it != stats->mostAccessedFiles.end()) {
          // Move to the end (most recently accessed)
          stats->mostAccessedFiles.erase(it);
          stats->mostAccessedFiles.push_back(record.path);
        } else {
          // Add to most accessed files, keep only top 10
          stats->mostAccessedFiles.push_back(record.path);
          if (stats->mostAccessedFiles.size() > 10) {
            stats->mostAccessedFiles.erase(stats->mostAccessedFiles.begin());
          }
        }
        
        // Calculate throughput (MB/s) - simplified, assumes uniform timing
        float durationSec = (now - stats->startTimestamp) / 1000.0f;
        if (durationSec > 0) {
          stats->readThroughputMBps = (stats->totalBytesRead / (1024.0f * 1024.0f)) / durationSec;
          stats->writeThroughputMBps = (stats->totalBytesWritten / (1024.0f * 1024.0f)) / durationSec;
        }
      },
      [&record, now]() {
        auto newStats = std::make_shared<WorkloadStats>();
        newStats->clientId = record.clientId;
        newStats->startTimestamp = now;
        newStats->lastActivityTimestamp = now;
        newStats->jobId = record.jobId;
        
        if (record.type == OperationType::READ) {
          newStats->readOps = 1;
          newStats->totalBytesRead = record.sizeBytes;
          newStats->randomReadOps = 1;
        } else if (record.type == OperationType::WRITE) {
          newStats->writeOps = 1;
          newStats->totalBytesWritten = record.sizeBytes;
        }
        
        if (record.latencyUs > 0) {
          newStats->avgLatencyUs = record.latencyUs;
          newStats->maxLatencyUs = record.latencyUs;
        }
        
        std::string ext = getFileExtension(record.path);
        if (!ext.empty()) {
          newStats->fileTypes[ext] = 1;
        }
        
        newStats->mostAccessedFiles.push_back(record.path);
        
        return newStats;
      });
}

std::shared_ptr<WorkloadStats> MLWorkloadAnalytics::getWorkloadStats(uint64_t clientId, const std::string& jobId) {
  auto it = activeWorkloads_.find(clientId);
  if (it == activeWorkloads_.end()) {
    return nullptr;
  }
  
  // If jobId is specified, check that it matches
  if (!jobId.empty() && it->second->jobId != jobId) {
    return nullptr;
  }
  
  return it->second;
}

std::unordered_map<uint64_t, std::shared_ptr<WorkloadStats>> MLWorkloadAnalytics::getAllWorkloadStats() {
  // Return a copy of the active workloads map
  return activeWorkloads_.getShallowCopies();
}

std::vector<std::shared_ptr<WorkloadStats>> MLWorkloadAnalytics::getHistoricalWorkloadStats(
    uint64_t startTime, uint64_t endTime) {
  std::vector<std::shared_ptr<WorkloadStats>> result;
  
  // Filter historical stats by time range
  for (const auto& stats : historicalStats_) {
    if (stats->startTimestamp <= endTime && stats->lastActivityTimestamp >= startTime) {
      result.push_back(stats);
    }
  }
  
  return result;
}

std::string MLWorkloadAnalytics::getWorkloadReport() {
  folly::dynamic report = folly::dynamic::object;
  folly::dynamic workloads = folly::dynamic::array;
  
  auto allStats = getAllWorkloadStats();
  int workloadCount = 0;  // Counter for number of workloads
  
  for (const auto& [clientId, stats] : allStats) {
    folly::dynamic workload = folly::dynamic::object;
    workload["client_id"] = clientId;
    workload["job_id"] = stats->jobId;
    workload["workload_type"] = workloadTypeToString(stats->type);
    workload["framework"] = frameworkTypeToString(stats->framework);
    workload["start_time"] = stats->startTimestamp;
    workload["last_activity"] = stats->lastActivityTimestamp;
    workload["duration_ms"] = stats->lastActivityTimestamp - stats->startTimestamp;
    workload["total_bytes_read"] = stats->totalBytesRead;
    workload["total_bytes_written"] = stats->totalBytesWritten;
    workload["read_ops"] = stats->readOps;
    workload["write_ops"] = stats->writeOps;
    workload["read_throughput_mbps"] = stats->readThroughputMBps;
    workload["write_throughput_mbps"] = stats->writeThroughputMBps;
    workload["avg_latency_us"] = stats->avgLatencyUs;
    workload["max_latency_us"] = stats->maxLatencyUs;
    
    folly::dynamic fileTypes = folly::dynamic::object;
    for (const auto& [ext, count] : stats->fileTypes) {
      fileTypes[ext] = count;
    }
    workload["file_types"] = std::move(fileTypes);
    
    folly::dynamic mostAccessed = folly::dynamic::array;
    for (const auto& file : stats->mostAccessedFiles) {
      mostAccessed.push_back(file);
    }
    workload["most_accessed_files"] = std::move(mostAccessed);
    
    workload["pattern"] = stats->detectedWorkloadPattern;
    
    workloads.push_back(std::move(workload));
    workloadCount++;  // Increment counter
  }
  
  report["timestamp"] = getCurrentTimeMs();
  report["workload_count"] = workloadCount;  // Use the counter instead of size()
  report["workloads"] = std::move(workloads);
  
  return folly::toJson(report);
}

void MLWorkloadAnalytics::runWorkloadAnalysis() {
  if (!config_.enabled()) {
    return;
  }

  try {
    // Clean up idle workloads
    cleanupIdleWorkloads();
    
    // Analyze active workloads
    analyzeWorkloads();
  } catch (const std::exception& ex) {
    std::cerr << "Exception during workload analysis: " << ex.what() << std::endl;
  }
}

void MLWorkloadAnalytics::analyzeWorkloads() {
  for (auto& [clientId, stats] : activeWorkloads_.getShallowCopies()) {
    // Only analyze workloads with sufficient operations
    if (stats->readOps + stats->writeOps < config_.min_ops_for_detection()) {
      continue;
    }
    
    // Detect workload type
    stats->type = detectWorkloadType(stats);
    
    // Detect framework type
    stats->framework = detectFrameworkType(stats);
    
    // Detect and set workload pattern
    auto operations = recentOperations_.find(clientId);
    if (operations != recentOperations_.end() && !operations->second.empty()) {
      // Analyze access patterns
      bool hasSequentialPattern = (stats->sequentialReadOps > stats->randomReadOps);
      bool hasBatchPattern = false;
      bool hasCheckpointPattern = false;
      
      // Check for checkpoint pattern (periodic large writes)
      if (stats->writeOps > 0) {
        std::vector<uint64_t> writeTimes;
        for (const auto& op : operations->second) {
          if (op.type == OperationType::WRITE && op.sizeBytes > 1024 * 1024) {
            writeTimes.push_back(op.timestamp);
          }
        }
        
        if (writeTimes.size() >= 3) {
          std::vector<uint64_t> intervals;
          for (size_t i = 1; i < writeTimes.size(); i++) {
            intervals.push_back(writeTimes[i] - writeTimes[i-1]);
          }
          
          // Check if intervals are approximately consistent
          uint64_t sum = std::accumulate(intervals.begin(), intervals.end(), 0ULL);
          uint64_t avg = sum / intervals.size();
          
          int consistentIntervals = 0;
          for (uint64_t interval : intervals) {
            if (std::abs(static_cast<int64_t>(interval - avg)) < avg * 0.2) {
              consistentIntervals++;
            }
          }
          
          if (consistentIntervals > intervals.size() * 0.7) {
            hasCheckpointPattern = true;
          }
        }
      }
      
      // Check for batch pattern (grouped reads)
      if (stats->readOps > 10) {
        std::vector<std::pair<uint64_t, uint64_t>> readBatches;  // start time, end time
        uint64_t batchStart = 0;
        uint64_t lastRead = 0;
        
        for (const auto& op : operations->second) {
          if (op.type == OperationType::READ) {
            if (batchStart == 0) {
              batchStart = op.timestamp;
              lastRead = op.timestamp;
            } else if (op.timestamp - lastRead > 500) {  // 500ms gap indicates new batch
              readBatches.emplace_back(batchStart, lastRead);
              batchStart = op.timestamp;
            }
            lastRead = op.timestamp;
          }
        }
        
        if (batchStart != 0) {
          readBatches.emplace_back(batchStart, lastRead);
        }
        
        if (readBatches.size() >= 3) {
          hasBatchPattern = true;
        }
      }
      
      // Set detected pattern
      std::string pattern;
      if (hasCheckpointPattern) {
        pattern = "Periodic checkpointing";
      } else if (hasBatchPattern && hasSequentialPattern) {
        pattern = "Sequential batch processing";
      } else if (hasBatchPattern) {
        pattern = "Batch random access";
      } else if (hasSequentialPattern) {
        pattern = "Sequential streaming";
      } else {
        pattern = "Random access";
      }
      
      stats->detectedWorkloadPattern = pattern;
      std::cout << "Detected pattern for client " << clientId << ": " << pattern << std::endl;
    }
  }
}

void MLWorkloadAnalytics::generateReport() {
  if (!config_.enabled()) {
    return;
  }

  std::cout << "Generating ML workload report" << std::endl;
  
  try {
    // Generate a report for all active workloads
    auto now = getCurrentTimeMs();
    std::vector<std::shared_ptr<WorkloadStats>> statsToStore;
    
    for (const auto& [clientId, stats] : activeWorkloads_.getShallowCopies()) {
      // Only include workloads with sufficient activity
      if (stats->readOps + stats->writeOps >= config_.min_ops_for_detection()) {
        // Create a copy for historical storage
        auto statsCopy = std::make_shared<WorkloadStats>(*stats);
        statsToStore.push_back(statsCopy);
        historicalStats_.push_back(statsCopy);
      }
    }
    
    // Store the stats in the database
    if (!statsToStore.empty()) {
      auto result = storeWorkloadStats(statsToStore);
      if (!result) {
        std::cerr << "Failed to store workload stats: " << result.error() << std::endl;
      }
    }
    
    // Trim historical stats to keep only the configured retention period
    uint64_t retentionMs = config_.history_retention_hours() * 3600 * 1000;
    uint64_t cutoffTime = now - retentionMs;
    
    historicalStats_.erase(
        std::remove_if(historicalStats_.begin(), historicalStats_.end(),
            [cutoffTime](const std::shared_ptr<WorkloadStats>& stats) {
              return stats->lastActivityTimestamp < cutoffTime;
            }),
        historicalStats_.end());
    
    std::cout << "ML workload report completed. Active workloads: " << activeWorkloads_.size()
               << ", Historical records: " << historicalStats_.size() << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "Exception during workload report generation: " << ex.what() << std::endl;
  }
}

Result<Void> MLWorkloadAnalytics::storeWorkloadStats(const std::vector<std::shared_ptr<WorkloadStats>>& stats) {
  // In a real implementation, this would connect to ClickHouse and store the data
  // For now, just log that we would store these stats
  std::cout << "Would store " << stats.size() << " workload stats records in database" << std::endl;
  
  // Format would be something like:
  // INSERT INTO ml_workload_stats (timestamp, client_id, job_id, workload_type, framework, ...)
  // VALUES (...)
  
  return success();
}

WorkloadType MLWorkloadAnalytics::detectWorkloadType(const std::shared_ptr<WorkloadStats>& stats) {
  // A real implementation would use more sophisticated heuristics
  // This is a simplified version based on operation ratios and file types
  
  // Check file extensions for clues
  bool hasModelFiles = false;
  bool hasDatasetFiles = false;
  bool hasCheckpointFiles = false;
  
  for (const auto& [ext, count] : stats->fileTypes) {
    if (ext == ".pt" || ext == ".pth" || ext == ".onnx" || ext == ".pb" || ext == ".h5") {
      hasModelFiles = true;
    }
    if (ext == ".tfrecord" || ext == ".mindrecord" || ext == ".coco" || ext == ".voc" || ext == ".csv") {
      hasDatasetFiles = true;
    }
    if (ext == ".ckpt" || ext == ".checkpoint") {
      hasCheckpointFiles = true;
    }
  }
  
  // Check file paths for clues
  bool hasTrainingPaths = false;
  bool hasInferencePaths = false;
  
  for (const auto& file : stats->mostAccessedFiles) {
    if (file.find("train") != std::string::npos || file.find("Train") != std::string::npos) {
      hasTrainingPaths = true;
    }
    if (file.find("infer") != std::string::npos || file.find("Infer") != std::string::npos ||
        file.find("predict") != std::string::npos || file.find("Predict") != std::string::npos) {
      hasInferencePaths = true;
    }
  }
  
  // Training workloads typically have both high read and write rates
  if ((stats->readOps > 100 && stats->writeOps > 10) || 
      (hasTrainingPaths && !hasInferencePaths) ||
      (hasModelFiles && hasDatasetFiles)) {
    return WorkloadType::TRAINING;
  }
  
  // Inference workloads typically have high read but low write rates
  if ((stats->readOps > 100 && stats->writeOps < 10) || 
      (hasInferencePaths && !hasTrainingPaths) ||
      (hasModelFiles && !hasDatasetFiles)) {
    return WorkloadType::INFERENCE;
  }
  
  // Checkpoint workloads have periodic large writes
  if (hasCheckpointFiles || hasCheckpointPattern(stats)) {
    return WorkloadType::CHECKPOINT;
  }
  
  // Data preparation workloads have more varied IO patterns
  if (hasDatasetFiles && !hasModelFiles) {
    return WorkloadType::DATA_PREP;
  }
  
  // Evaluation workloads look similar to inference but may write results
  if (stats->readOps > 10 && stats->writeOps > 0 && 
      stats->writeOps < stats->readOps * 0.2) {
    return WorkloadType::EVALUATION;
  }
  
  return WorkloadType::UNKNOWN;
}

bool MLWorkloadAnalytics::hasCheckpointPattern(const std::shared_ptr<WorkloadStats>& stats) {
  auto clientId = stats->clientId;
  auto ops = recentOperations_.find(clientId);
  if (ops == recentOperations_.end()) {
    return false;
  }
  
  // Look for periodic large writes
  const auto& operations = ops->second;
  std::vector<uint64_t> largeWriteTimes;
  
  for (const auto& op : operations) {
    if (op.type == OperationType::WRITE && op.sizeBytes > 1024 * 1024) {
      largeWriteTimes.push_back(op.timestamp);
    }
  }
  
  if (largeWriteTimes.size() < 3) {
    return false;
  }
  
  // Check for periodicity
  std::vector<uint64_t> intervals;
  for (size_t i = 1; i < largeWriteTimes.size(); i++) {
    intervals.push_back(largeWriteTimes[i] - largeWriteTimes[i-1]);
  }
  
  uint64_t sum = std::accumulate(intervals.begin(), intervals.end(), 0ULL);
  uint64_t avg = sum / intervals.size();
  
  // Count intervals that are close to the average
  int consistentIntervals = 0;
  for (uint64_t interval : intervals) {
    if (std::abs(static_cast<int64_t>(interval - avg)) < avg * 0.25) {
      consistentIntervals++;
    }
  }
  
  // If most intervals are consistent, it's likely a checkpoint pattern
  return consistentIntervals >= intervals.size() * 0.7;
}

FrameworkType MLWorkloadAnalytics::detectFrameworkType(const std::shared_ptr<WorkloadStats>& stats) {
  // Count occurrences of framework-specific file extensions and path patterns
  int pytorchScore = 0;
  int tensorflowScore = 0;
  int jaxScore = 0;
  int mxnetScore = 0;
  
  // Check file extensions
  for (const auto& [ext, count] : stats->fileTypes) {
    if (ext == ".pt" || ext == ".pth") {
      pytorchScore += count;
    } else if (ext == ".pb" || ext == ".tfrecord" || ext == ".index") {
      tensorflowScore += count;
    } else if (ext == ".npy") {
      jaxScore += count / 2;  // JAX often uses numpy, but not exclusively
    } else if (ext == ".params") {
      mxnetScore += count;
    }
  }
  
  // Check file paths for framework-specific patterns
  for (const auto& file : stats->mostAccessedFiles) {
    std::string lowerFile = file;
    std::transform(lowerFile.begin(), lowerFile.end(), lowerFile.begin(), ::tolower);
    
    if (lowerFile.find("torch") != std::string::npos || lowerFile.find("pytorch") != std::string::npos) {
      pytorchScore += 5;
    } else if (lowerFile.find("tensorflow") != std::string::npos || lowerFile.find("tf_") != std::string::npos) {
      tensorflowScore += 5;
    } else if (lowerFile.find("jax") != std::string::npos || lowerFile.find("flax") != std::string::npos) {
      jaxScore += 5;
    } else if (lowerFile.find("mxnet") != std::string::npos) {
      mxnetScore += 5;
    }
  }
  
  // Detect the most likely framework
  int maxScore = std::max({pytorchScore, tensorflowScore, jaxScore, mxnetScore});
  
  if (maxScore == 0) {
    return FrameworkType::UNKNOWN;
  } else if (maxScore == pytorchScore) {
    return FrameworkType::PYTORCH;
  } else if (maxScore == tensorflowScore) {
    return FrameworkType::TENSORFLOW;
  } else if (maxScore == jaxScore) {
    return FrameworkType::JAX;
  } else if (maxScore == mxnetScore) {
    return FrameworkType::MXNET;
  } else {
    return FrameworkType::CUSTOM;
  }
}

void MLWorkloadAnalytics::cleanupIdleWorkloads() {
  auto now = getCurrentTimeMs();
  uint64_t idleThreshold = config_.workload_idle_timeout_ms();
  std::vector<uint64_t> clientsToRemove;
  
  // Find idle workloads
  for (const auto& [clientId, stats] : activeWorkloads_.getShallowCopies()) {
    if (now - stats->lastActivityTimestamp > idleThreshold) {
      clientsToRemove.push_back(clientId);
      
      // Add to historical stats before removing if it has significant activity
      if (stats->readOps + stats->writeOps >= config_.min_ops_for_detection()) {
        auto statsCopy = std::make_shared<WorkloadStats>(*stats);
        historicalStats_.push_back(statsCopy);
      }
    }
  }
  
  // Remove idle workloads
  for (uint64_t clientId : clientsToRemove) {
    activeWorkloads_.erase(clientId);
    recentOperations_.erase(clientId);
    std::cout << "Removed idle workload for client " << clientId << std::endl;
  }
  
  // Ensure we don't exceed the maximum number of tracked workloads
  if (activeWorkloads_.size() > config_.max_workloads_to_track()) {
    std::cout << "Too many active workloads (" << activeWorkloads_.size() 
               << "), removing some" << std::endl;
    
    // Find the least active workloads
    std::vector<std::pair<uint64_t, uint64_t>> activityLevels;
    for (const auto& [clientId, stats] : activeWorkloads_.getShallowCopies()) {
      activityLevels.emplace_back(clientId, stats->readOps + stats->writeOps);
    }
    
    // Sort by activity (ascending)
    std::sort(activityLevels.begin(), activityLevels.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Remove the least active workloads
    size_t removeCount = activeWorkloads_.size() - config_.max_workloads_to_track();
    for (size_t i = 0; i < removeCount && i < activityLevels.size(); i++) {
      uint64_t clientId = activityLevels[i].first;
      activeWorkloads_.erase(clientId);
      recentOperations_.erase(clientId);
    }
  }
}

} // namespace hf3fs::monitor_collector 