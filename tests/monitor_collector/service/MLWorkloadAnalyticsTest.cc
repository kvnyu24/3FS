#include "monitor_collector/service/MLWorkloadAnalytics.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/test/TestUtils.h>
#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace hf3fs::monitor_collector {

namespace {

// Helper to create an operation record 
OperationRecord createOperationRecord(
    OperationType type,
    uint64_t clientId,
    const std::string& path,
    uint64_t size,
    uint64_t latency,
    uint64_t offset = 0,
    const std::string& jobId = "test-job") {
  
  OperationRecord record;
  record.type = type;
  record.clientId = clientId;
  record.path = path;
  record.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
  record.sizeBytes = size;
  record.latencyUs = latency;
  record.offset = offset;
  record.jobId = jobId;
  
  return record;
}

} // namespace

class MLWorkloadAnalyticsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create config with reasonable test values
    MLWorkloadAnalytics::Config config;
    config.set_enabled(true);
    config.set_analysis_interval_ms(100);
    config.set_report_interval_ms(500);
    config.set_min_ops_for_detection(10);
    config.set_max_workloads_to_track(100);
    config.set_workload_idle_timeout_ms(60000);
    config.set_history_retention_hours(24);
    
    // Create the analytics service
    analytics_ = std::make_unique<MLWorkloadAnalytics>(config);
    
    // Initialize the service
    EXPECT_TRUE(analytics_->init());
  }
  
  void TearDown() override {
    auto result = analytics_->stopAndJoin();
    EXPECT_TRUE(result);
    analytics_.reset();
  }
  
  // Helper to simulate ML training workload
  void simulateTrainingWorkload(uint64_t clientId) {
    // Training typically has:
    // 1. Dataset reads (often sequential batches)
    // 2. Periodic model checkpoints (writes)
    // 3. Consistent patterns over time
    
    // First, simulate dataset loading
    for (int i = 0; i < 20; i++) {
      std::string path = "/datasets/imagenet/batch_" + std::to_string(i) + ".tfrecord";
      auto record = createOperationRecord(
          OperationType::READ, clientId, path, 10 * 1024 * 1024, 15000, i * 10 * 1024 * 1024);
      analytics_->recordOperation(record);
    }
    
    // Now simulate checkpoint writes
    for (int epoch = 0; epoch < 5; epoch++) {
      // Do some processing...
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      
      // Then checkpoint
      std::string checkpointPath = "/models/checkpoint_" + std::to_string(epoch) + ".pt";
      auto record = createOperationRecord(
          OperationType::WRITE, clientId, checkpointPath, 500 * 1024 * 1024, 200000);
      analytics_->recordOperation(record);
    }
  }
  
  // Helper to simulate ML inference workload
  void simulateInferenceWorkload(uint64_t clientId) {
    // Inference typically has:
    // 1. Model file reads (once at startup)
    // 2. Small input data reads
    // 3. Optional small writes for results
    
    // Model loading
    auto modelRecord = createOperationRecord(
        OperationType::READ, clientId, "/models/resnet50.pt", 100 * 1024 * 1024, 50000);
    analytics_->recordOperation(modelRecord);
    
    // Process many small inputs
    for (int i = 0; i < 30; i++) {
      std::string inputPath = "/data/inputs/image_" + std::to_string(i) + ".jpg";
      auto inputRecord = createOperationRecord(
          OperationType::READ, clientId, inputPath, 200 * 1024, 5000);
      analytics_->recordOperation(inputRecord);
      
      // Occasionally write results
      if (i % 5 == 0) {
        std::string resultPath = "/data/results/result_" + std::to_string(i) + ".json";
        auto resultRecord = createOperationRecord(
            OperationType::WRITE, clientId, resultPath, 10 * 1024, 2000);
        analytics_->recordOperation(resultRecord);
      }
    }
  }
  
  // Helper to simulate data preparation workload
  void simulateDataPrepWorkload(uint64_t clientId) {
    // Data prep typically has:
    // 1. Many reads from raw data
    // 2. Processing
    // 3. Writes to processed dataset files
    
    // Read raw data
    for (int i = 0; i < 40; i++) {
      std::string rawPath = "/raw_data/file_" + std::to_string(i) + ".csv";
      auto readRecord = createOperationRecord(
          OperationType::READ, clientId, rawPath, 5 * 1024 * 1024, 10000);
      analytics_->recordOperation(readRecord);
    }
    
    // Write processed data
    for (int i = 0; i < 10; i++) {
      std::string processedPath = "/processed_data/batch_" + std::to_string(i) + ".tfrecord";
      auto writeRecord = createOperationRecord(
          OperationType::WRITE, clientId, processedPath, 20 * 1024 * 1024, 30000);
      analytics_->recordOperation(writeRecord);
    }
  }
  
  std::unique_ptr<MLWorkloadAnalytics> analytics_;
};

TEST_F(MLWorkloadAnalyticsTest, DetectsWorkloadTypes) {
  // Simulate different workloads with different client IDs
  simulateTrainingWorkload(1001);
  simulateInferenceWorkload(1002);
  simulateDataPrepWorkload(1003);
  
  // Let the analysis run
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  
  // Check detection results
  auto trainingStats = analytics_->getWorkloadStats(1001);
  ASSERT_NE(trainingStats, nullptr);
  EXPECT_EQ(trainingStats->type, WorkloadType::TRAINING);
  
  auto inferenceStats = analytics_->getWorkloadStats(1002);
  ASSERT_NE(inferenceStats, nullptr);
  EXPECT_EQ(inferenceStats->type, WorkloadType::INFERENCE);
  
  auto dataPrepStats = analytics_->getWorkloadStats(1003);
  ASSERT_NE(dataPrepStats, nullptr);
  EXPECT_EQ(dataPrepStats->type, WorkloadType::DATA_PREP);
}

TEST_F(MLWorkloadAnalyticsTest, TracksAccessPatterns) {
  uint64_t clientId = 2001;
  
  // Create a sequential access pattern
  for (int i = 0; i < 20; i++) {
    std::string path = "/data/file.bin";
    uint64_t offset = i * 1024 * 1024;
    auto record = createOperationRecord(
        OperationType::READ, clientId, path, 1024 * 1024, 5000, offset);
    analytics_->recordOperation(record);
  }
  
  // Let the analysis run
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  
  // Check the detected pattern
  auto stats = analytics_->getWorkloadStats(clientId);
  ASSERT_NE(stats, nullptr);
  EXPECT_GT(stats->sequentialReadOps, stats->randomReadOps);
  EXPECT_TRUE(stats->detectedWorkloadPattern.find("Sequential") != std::string::npos);
}

TEST_F(MLWorkloadAnalyticsTest, GeneratesReport) {
  // Simulate multiple workloads
  simulateTrainingWorkload(3001);
  simulateInferenceWorkload(3002);
  
  // Let the analysis and reporting run
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  
  // Get the report
  std::string report = analytics_->getWorkloadReport();
  
  // Verify report has expected content
  EXPECT_FALSE(report.empty());
  EXPECT_NE(report.find("workload_count"), std::string::npos);
  EXPECT_NE(report.find("3001"), std::string::npos);
  EXPECT_NE(report.find("3002"), std::string::npos);
}

TEST_F(MLWorkloadAnalyticsTest, DetectsFrameworkType) {
  uint64_t clientId = 4001;
  
  // Simulate PyTorch usage
  analytics_->recordOperation(createOperationRecord(
      OperationType::READ, clientId, "/models/model.pt", 100 * 1024 * 1024, 50000));
  analytics_->recordOperation(createOperationRecord(
      OperationType::WRITE, clientId, "/checkpoints/checkpoint.pth", 200 * 1024 * 1024, 100000));
  analytics_->recordOperation(createOperationRecord(
      OperationType::READ, clientId, "/pytorch/datasets/data.bin", 50 * 1024 * 1024, 20000));
  
  // Let the analysis run
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  
  // Check framework detection
  auto stats = analytics_->getWorkloadStats(clientId);
  ASSERT_NE(stats, nullptr);
  EXPECT_EQ(stats->framework, FrameworkType::PYTORCH);
  
  // Now test TensorFlow detection
  uint64_t clientId2 = 4002;
  analytics_->recordOperation(createOperationRecord(
      OperationType::READ, clientId2, "/tensorflow/model.pb", 100 * 1024 * 1024, 50000));
  analytics_->recordOperation(createOperationRecord(
      OperationType::READ, clientId2, "/data/train.tfrecord", 200 * 1024 * 1024, 100000));
  
  // Let the analysis run
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  
  // Check framework detection
  auto stats2 = analytics_->getWorkloadStats(clientId2);
  ASSERT_NE(stats2, nullptr);
  EXPECT_EQ(stats2->framework, FrameworkType::TENSORFLOW);
}

TEST_F(MLWorkloadAnalyticsTest, HistoricalStats) {
  // Create a client with some activity
  uint64_t clientId = 5001;
  simulateTrainingWorkload(clientId);
  
  // Let the analysis and reporting run
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  
  // Force the client to become idle and be moved to historical stats
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  auto hourAgo = now - (3600 * 1000);
  
  // Get historical stats for the last hour
  auto historicalStats = analytics_->getHistoricalWorkloadStats(hourAgo, now);
  
  // We should have at least one historical record
  EXPECT_FALSE(historicalStats.empty());
}

TEST_F(MLWorkloadAnalyticsTest, CleanupIdleWorkloads) {
  // Create a client with minimal activity
  uint64_t clientId = 6001;
  analytics_->recordOperation(createOperationRecord(
      OperationType::READ, clientId, "/data/file.txt", 1024, 1000));
  
  // Check that it's in active workloads
  auto stats = analytics_->getWorkloadStats(clientId);
  ASSERT_NE(stats, nullptr);
  
  // Modify the idle timeout to be very short for testing
  MLWorkloadAnalytics::Config config;
  config.set_enabled(true);
  config.set_workload_idle_timeout_ms(100);  // Very short timeout
  
  // Create a new analytics service with the short timeout
  auto analytics = std::make_unique<MLWorkloadAnalytics>(config);
  EXPECT_TRUE(analytics->init());
  
  // Add the same client
  analytics->recordOperation(createOperationRecord(
      OperationType::READ, clientId, "/data/file.txt", 1024, 1000));
  
  // Wait for cleanup to run
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  
  // Client should be removed due to inactivity
  auto statsAfterCleanup = analytics->getWorkloadStats(clientId);
  EXPECT_EQ(statsAfterCleanup, nullptr);
  
  analytics->stopAndJoin();
}

} // namespace hf3fs::monitor_collector 