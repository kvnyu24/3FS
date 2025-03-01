#include "mgmtd/service/StorageMgmt.h"

#include <folly/dynamic.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>

namespace hf3fs::mgmtd {

StorageMgmt::StorageMgmt(std::shared_ptr<storage::StorageClient> storageClient)
    : storageClient_(storageClient) {
  // Initialize the ML Workload Analytics service
  monitor_collector::MLWorkloadAnalytics::Config config;
  config.set_enabled(true);
  mlWorkloadAnalytics_ = std::make_shared<monitor_collector::MLWorkloadAnalytics>(config);
  mlWorkloadAnalytics_->init();
}

// ... existing methods ...

CoTask<MgmtResponse> StorageMgmt::handleMLWorkloadAnalytics(const MgmtRequest& request) {
  MgmtResponse response;
  response.status = MgmtStatus::OK;
  
  try {
    // Get query parameters
    std::string operation = "report";  // Default operation
    std::string clientIdStr;
    std::string jobId;
    std::string startTimeStr;
    std::string endTimeStr;
    
    if (request.has_params()) {
      auto params = request.params();
      auto it = params.find("operation");
      if (it != params.end()) {
        operation = it->second;
      }
      
      it = params.find("client_id");
      if (it != params.end()) {
        clientIdStr = it->second;
      }
      
      it = params.find("job_id");
      if (it != params.end()) {
        jobId = it->second;
      }
      
      it = params.find("start_time");
      if (it != params.end()) {
        startTimeStr = it->second;
      }
      
      it = params.find("end_time");
      if (it != params.end()) {
        endTimeStr = it->second;
      }
    }
    
    folly::dynamic result = folly::dynamic::object;
    
    if (operation == "report") {
      // Get the full workload report
      std::string report = mlWorkloadAnalytics_->getWorkloadReport();
      try {
        result = folly::parseJson(report);
      } catch (const std::exception& ex) {
        XLOG(ERR) << "Failed to parse workload report: " << ex.what();
        result["error"] = "Failed to parse workload report";
        response.status = MgmtStatus::INTERNAL_ERROR;
      }
    } else if (operation == "client") {
      // Get stats for a specific client
      if (clientIdStr.empty()) {
        result["error"] = "Missing client_id parameter";
        response.status = MgmtStatus::BAD_REQUEST;
      } else {
        try {
          uint64_t clientId = std::stoull(clientIdStr);
          auto stats = mlWorkloadAnalytics_->getWorkloadStats(clientId, jobId);
          
          if (stats) {
            result["client_id"] = clientId;
            result["job_id"] = stats->jobId;
            result["workload_type"] = static_cast<int>(stats->type);
            result["framework"] = static_cast<int>(stats->framework);
            result["start_timestamp"] = stats->startTimestamp;
            result["last_activity_timestamp"] = stats->lastActivityTimestamp;
            result["total_bytes_read"] = stats->totalBytesRead;
            result["total_bytes_written"] = stats->totalBytesWritten;
            result["read_ops"] = stats->readOps;
            result["write_ops"] = stats->writeOps;
            result["sequential_read_ops"] = stats->sequentialReadOps;
            result["random_read_ops"] = stats->randomReadOps;
            result["read_throughput_mbps"] = stats->readThroughputMBps;
            result["write_throughput_mbps"] = stats->writeThroughputMBps;
            result["avg_latency_us"] = stats->avgLatencyUs;
            result["max_latency_us"] = stats->maxLatencyUs;
            result["detected_pattern"] = stats->detectedWorkloadPattern;
            
            folly::dynamic fileTypes = folly::dynamic::object;
            for (const auto& [ext, count] : stats->fileTypes) {
              fileTypes[ext] = count;
            }
            result["file_types"] = std::move(fileTypes);
            
            folly::dynamic mostAccessed = folly::dynamic::array;
            for (const auto& file : stats->mostAccessedFiles) {
              mostAccessed.push_back(file);
            }
            result["most_accessed_files"] = std::move(mostAccessed);
          } else {
            result["error"] = "Client not found";
            response.status = MgmtStatus::NOT_FOUND;
          }
        } catch (const std::exception& ex) {
          result["error"] = "Invalid client_id parameter";
          response.status = MgmtStatus::BAD_REQUEST;
        }
      }
    } else if (operation == "historical") {
      // Get historical stats for a time range
      uint64_t startTime = 0;
      uint64_t endTime = std::numeric_limits<uint64_t>::max();
      
      if (!startTimeStr.empty()) {
        try {
          startTime = std::stoull(startTimeStr);
        } catch (const std::exception&) {
          result["error"] = "Invalid start_time parameter";
          response.status = MgmtStatus::BAD_REQUEST;
          response.body = folly::toJson(result);
          co_return response;
        }
      }
      
      if (!endTimeStr.empty()) {
        try {
          endTime = std::stoull(endTimeStr);
        } catch (const std::exception&) {
          result["error"] = "Invalid end_time parameter";
          response.status = MgmtStatus::BAD_REQUEST;
          response.body = folly::toJson(result);
          co_return response;
        }
      }
      
      auto historicalStats = mlWorkloadAnalytics_->getHistoricalWorkloadStats(startTime, endTime);
      folly::dynamic stats = folly::dynamic::array;
      
      for (const auto& stat : historicalStats) {
        folly::dynamic entry = folly::dynamic::object;
        entry["client_id"] = stat->clientId;
        entry["job_id"] = stat->jobId;
        entry["workload_type"] = static_cast<int>(stat->type);
        entry["framework"] = static_cast<int>(stat->framework);
        entry["start_timestamp"] = stat->startTimestamp;
        entry["last_activity_timestamp"] = stat->lastActivityTimestamp;
        entry["total_bytes_read"] = stat->totalBytesRead;
        entry["total_bytes_written"] = stat->totalBytesWritten;
        entry["read_ops"] = stat->readOps;
        entry["write_ops"] = stat->writeOps;
        entry["detected_pattern"] = stat->detectedWorkloadPattern;
        
        stats.push_back(std::move(entry));
      }
      
      result["historical_stats"] = std::move(stats);
      result["count"] = historicalStats.size();
      result["start_time"] = startTime;
      result["end_time"] = endTime;
    } else {
      result["error"] = "Invalid operation parameter";
      response.status = MgmtStatus::BAD_REQUEST;
    }
    
    response.body = folly::toJson(result);
  } catch (const std::exception& ex) {
    XLOG(ERR) << "Exception in handleMLWorkloadAnalytics: " << ex.what();
    response.status = MgmtStatus::INTERNAL_ERROR;
    response.body = folly::toJson(folly::dynamic::object("error", ex.what()));
  }
  
  co_return response;
}

} // namespace hf3fs::mgmtd 