#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

// We'll implement these directly to avoid dependency issues
namespace folly {
  template <typename K, typename V>
  class ConcurrentHashMap {
  public:
    using const_iterator = typename std::unordered_map<K, V>::const_iterator;

    V& operator[](const K& key) {
      return data_[key];
    }
    
    typename std::unordered_map<K, V>::iterator find(const K& key) {
      return data_.find(key);
    }
    
    typename std::unordered_map<K, V>::iterator begin() {
      return data_.begin();
    }
    
    typename std::unordered_map<K, V>::iterator end() {
      return data_.end();
    }
    
    typename std::unordered_map<K, V>::const_iterator end() const {
      return data_.end();
    }
    
    size_t size() const {
      return data_.size();
    }
    
    void erase(const K& key) {
      data_.erase(key);
    }
    
    template <typename F>
    void upsert(const K& key, F&& fn, std::function<V()> factory) {
      auto it = data_.find(key);
      if (it != data_.end()) {
        fn(it->second);
      } else {
        data_[key] = factory();
      }
    }
    
    std::unordered_map<K, V> getShallowCopies() const {
      return data_;
    }
    
  private:
    std::unordered_map<K, V> data_;
  };
  
  // Simple implementation of folly::dynamic for testing
  class dynamic {
  public:
    static dynamic object;
    static dynamic array;
    
    dynamic() : type_(Type::OBJECT) {}
    dynamic(const char* str) : type_(Type::STRING), stringValue_(str) {}
    dynamic(const std::string& str) : type_(Type::STRING), stringValue_(str) {}
    dynamic(int val) : type_(Type::INT), intValue_(val) {}
    dynamic(unsigned int val) : type_(Type::INT), intValue_(val) {}
    dynamic(int64_t val) : type_(Type::INT), intValue_(val) {}
    dynamic(uint64_t val) : type_(Type::INT), intValue_(static_cast<int64_t>(val)) {}
    dynamic(double val) : type_(Type::DOUBLE), doubleValue_(val) {}
    dynamic(bool val) : type_(Type::BOOL), boolValue_(val) {}
    
    void push_back(dynamic val) {
      type_ = Type::ARRAY;
      arrayValue_.push_back(std::move(val));
    }
    
    dynamic& operator[](const std::string& key) {
      type_ = Type::OBJECT;
      return objectValue_[key];
    }
    
    dynamic& operator[](const char* key) {
      return (*this)[std::string(key)];
    }
    
    // Added to prevent warnings about unused fields
    std::string asString() const {
      switch (type_) {
        case Type::STRING: return stringValue_;
        case Type::INT: return std::to_string(intValue_);
        case Type::DOUBLE: return std::to_string(doubleValue_);
        case Type::BOOL: return boolValue_ ? "true" : "false";
        default: return "{}";
      }
    }
    
  private:
    enum class Type {
      OBJECT,
      ARRAY,
      STRING,
      INT,
      DOUBLE,
      BOOL
    };
    
    Type type_;
    std::unordered_map<std::string, dynamic> objectValue_;
    std::vector<dynamic> arrayValue_;
    std::string stringValue_;
    int64_t intValue_ = 0;
    double doubleValue_ = 0.0;
    bool boolValue_ = false;
  };
  
  // Stub for toJson function
  inline std::string toJson(const dynamic&) {
    return "{}"; // Return empty JSON for now
  }
  
  // Stub for parseJson function
  inline dynamic parseJson(const std::string&) {
    return dynamic::object;
  }
}

// Simple implementations of utils classes
namespace hf3fs {
  class Void {
  public:
    Void() = default;
  };
  
  template <typename T, typename E = std::string>
  class Result {
  public:
    Result(T value) : value_(std::move(value)), hasValue_(true) {}
    Result(E error) : error_(std::move(error)), hasValue_(false) {}
    
    bool hasValue() const { return hasValue_; }
    bool hasError() const { return !hasValue_; }
    
    const T& value() const& { return value_; }
    const E& error() const& { return error_; }
    
    operator bool() const { return hasValue_; }
    
  private:
    T value_;
    E error_;
    bool hasValue_;
  };
  
  inline Result<Void, std::string> success() {
    return Result<Void, std::string>(Void{});
  }
  
  class BackgroundRunner {
  public:
    BackgroundRunner(const std::string& name, std::function<void()> fn, std::chrono::milliseconds interval)
        : name_(name), fn_(std::move(fn)) {}
    
    void start() {}
    void stop() {}
    void join() {}
    
  private:
    std::string name_;
    std::function<void()> fn_;
  };
  
  namespace utils {
    class CallbackGuard {
    public:
      CallbackGuard() = default;
      CallbackGuard(std::function<void()> fn) : fn_(std::move(fn)) {}
      
    private:
      std::function<void()> fn_;
    };
  }
}

namespace hf3fs::monitor_collector {

/**
 * ML workload type classification 
 */
enum class WorkloadType {
  UNKNOWN,
  TRAINING,
  INFERENCE,
  DATA_PREP,
  EVALUATION,
  CHECKPOINT
};

/**
 * ML framework type
 */
enum class FrameworkType {
  UNKNOWN,
  PYTORCH,
  TENSORFLOW, 
  JAX,
  MXNET,
  CUSTOM
};

/**
 * Structure to store ML workload statistics
 */
struct WorkloadStats {
  WorkloadType type{WorkloadType::UNKNOWN};
  FrameworkType framework{FrameworkType::UNKNOWN};
  std::string jobId;
  uint64_t clientId{0};
  uint64_t startTimestamp{0};
  uint64_t lastActivityTimestamp{0};
  uint64_t totalBytesRead{0};
  uint64_t totalBytesWritten{0};
  uint64_t readOps{0};
  uint64_t writeOps{0};
  uint64_t randomReadOps{0};
  uint64_t sequentialReadOps{0};
  float readThroughputMBps{0.0};
  float writeThroughputMBps{0.0};
  uint64_t avgLatencyUs{0};
  uint64_t maxLatencyUs{0};
  std::unordered_map<std::string, uint64_t> fileTypes; // Extension to count
  std::vector<std::string> mostAccessedFiles;
  std::string detectedWorkloadPattern;
};

/**
 * API operation types to track
 */
enum class OperationType {
  READ,
  WRITE,
  OPEN,
  CLOSE,
  STAT,
  LIST,
  CREATE,
  MKDIR,
  REMOVE,
  RENAME
};

/**
 * Structure for a single operation record
 */
struct OperationRecord {
  OperationType type;
  uint64_t clientId;
  std::string path;
  uint64_t timestamp;
  uint64_t sizeBytes;
  uint64_t latencyUs;
  std::string jobId;
  uint64_t offset;
};

/**
 * ML Workload Analytics service for tracking ML workload interactions with the file system
 */
class MLWorkloadAnalytics {
public:
  class Config {
  public:
    // Methods to access config values
    bool enabled() const { return enabled_; }
    uint64_t analysis_interval_ms() const { return analysisIntervalMs_; }
    uint64_t report_interval_ms() const { return reportIntervalMs_; }
    uint64_t max_workloads_to_track() const { return maxWorkloadsToTrack_; }
    uint64_t min_ops_for_detection() const { return minOpsForDetection_; }
    uint64_t workload_idle_timeout_ms() const { return workloadIdleTimeoutMs_; }
    uint64_t history_retention_hours() const { return historyRetentionHours_; }
    
    // Methods to set config values
    void set_enabled(bool val) { enabled_ = val; }
    void set_analysis_interval_ms(uint64_t val) { analysisIntervalMs_ = val; }
    void set_report_interval_ms(uint64_t val) { reportIntervalMs_ = val; }
    void set_max_workloads_to_track(uint64_t val) { maxWorkloadsToTrack_ = val; }
    void set_min_ops_for_detection(uint64_t val) { minOpsForDetection_ = val; }
    void set_workload_idle_timeout_ms(uint64_t val) { workloadIdleTimeoutMs_ = val; }
    void set_history_retention_hours(uint64_t val) { historyRetentionHours_ = val; }
    
    // Callback registration method
    utils::CallbackGuard addCallbackGuard(std::function<void()> fn) {
      return utils::CallbackGuard(fn);
    }
    
  private:
    bool enabled_ = true;
    uint64_t analysisIntervalMs_ = 60000;    // 1 minute
    uint64_t reportIntervalMs_ = 300000;     // 5 minutes
    uint64_t maxWorkloadsToTrack_ = 1000;
    uint64_t minOpsForDetection_ = 100;
    uint64_t workloadIdleTimeoutMs_ = 600000;  // 10 minutes
    uint64_t historyRetentionHours_ = 24;
  };

  explicit MLWorkloadAnalytics(const Config& config);
  ~MLWorkloadAnalytics();

  /**
   * Initialize the analytics service
   */
  Result<Void> init();
  
  /**
   * Stop the analytics service
   */
  Result<Void> stopAndJoin();
  
  /**
   * Record a file system operation for analysis
   * 
   * @param record Operation details to record
   */
  void recordOperation(const OperationRecord& record);
  
  /**
   * Get current stats for a specific workload
   * 
   * @param clientId ID of the client to get stats for
   * @param jobId Optional job ID to filter by
   * @return The workload statistics or null if not found
   */
  std::shared_ptr<WorkloadStats> getWorkloadStats(uint64_t clientId, const std::string& jobId = "");
  
  /**
   * Get all current workload stats
   * 
   * @return Map of client IDs to their workload statistics
   */
  std::unordered_map<uint64_t, std::shared_ptr<WorkloadStats>> getAllWorkloadStats();
  
  /**
   * Get workload stats for a specific time range
   * 
   * @param startTime Start timestamp
   * @param endTime End timestamp
   * @return Vector of workload statistics in the specified time range
   */
  std::vector<std::shared_ptr<WorkloadStats>> getHistoricalWorkloadStats(
      uint64_t startTime, uint64_t endTime);
  
  /**
   * Get a JSON report of current workloads
   * 
   * @return JSON string containing workload analytics
   */
  std::string getWorkloadReport();

  /**
   * For testing: give access to internal data structures
   */
  folly::ConcurrentHashMap<uint64_t, std::vector<OperationRecord>> recentOperations_;
  folly::ConcurrentHashMap<uint64_t, std::shared_ptr<WorkloadStats>> activeWorkloads_;

private:
  /**
   * Run periodic workload analysis
   */
  void runWorkloadAnalysis();
  
  /**
   * Perform the actual workload analysis logic
   */
  void analyzeWorkloads();
  
  /**
   * Generate a workload report and store in database
   */
  void generateReport();
  
  /**
   * Store workload stats in ClickHouse database
   */
  Result<Void> storeWorkloadStats(const std::vector<std::shared_ptr<WorkloadStats>>& stats);
  
  /**
   * Detect workload type based on access patterns
   */
  WorkloadType detectWorkloadType(const std::shared_ptr<WorkloadStats>& stats);
  
  /**
   * Helper method to check if a workload has checkpoint pattern
   */
  bool hasCheckpointPattern(const std::shared_ptr<WorkloadStats>& stats);
  
  /**
   * Detect ML framework based on access patterns
   */
  FrameworkType detectFrameworkType(const std::shared_ptr<WorkloadStats>& stats);
  
  /**
   * Clean up idle workloads
   */
  void cleanupIdleWorkloads();

  Config config_;
  
  // Historical workload reports stored in memory
  std::vector<std::shared_ptr<WorkloadStats>> historicalStats_;
  
  // Background runners
  std::unique_ptr<BackgroundRunner> analysisRunner_;
  std::unique_ptr<BackgroundRunner> reportRunner_;
  
  // Configuration change callback
  utils::CallbackGuard onConfigUpdated_;
};

} // namespace hf3fs::monitor_collector 