#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

#include "common/utils/Coroutine.h"
#include "common/utils/Result.h"
#include "storage/service/Components.h"
#include "storage/service/PrefetchManager.h"
#include "storage/service/StorageOperator.h"
#include "storage/store/Chunk.h"

namespace hf3fs::storage {

class MockStorageOperator : public StorageOperator {
public:
  MockStorageOperator(const StorageOperator::Config& config, Components& components)
      : StorageOperator(config, components) {}
      
  // Mocked method for prefetching chunks
  MOCK_METHOD(CoTryTask<bool>, prefetchChunk, (const ChunkId&), ());
};

class PrefetchManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Set up components
    storageOperatorConfig = std::make_unique<StorageOperator::Config>();
    mockStorageOperator = std::make_unique<MockStorageOperator>(*storageOperatorConfig, components);
    
    // Configure prefetch manager
    PrefetchManager::Config config;
    config.enabled() = true;
    config.history_length() = 16;
    config.prefetch_threshold() = 0.6f;
    config.max_prefetch_chunks() = 8;
    config.prefetch_check_interval_ms() = 100;
    
    // Create the prefetch manager
    prefetchManager = std::make_unique<PrefetchManager>(config, components, *mockStorageOperator);
  }

  void TearDown() override {
    if (prefetchManager) {
      prefetchManager->stopAndJoin();
    }
  }

  // Helper to create sequential access pattern
  void recordSequentialAccess(uint64_t clientId, uint64_t startChunk, int count) {
    for (int i = 0; i < count; i++) {
      ChunkId chunkId(startChunk + i);
      prefetchManager->recordAccess(chunkId, clientId, 0, 1024);
    }
  }
  
  // Helper to create strided access pattern
  void recordStridedAccess(uint64_t clientId, uint64_t startChunk, int stride, int count) {
    for (int i = 0; i < count; i++) {
      ChunkId chunkId(startChunk + (i * stride));
      prefetchManager->recordAccess(chunkId, clientId, 0, 1024);
    }
  }
  
  // Helper to create transpose-like access pattern
  void recordTransposeAccess(uint64_t clientId, uint64_t startChunk, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        // Create a pattern like accessing elements in a 2D array but in column-major order
        ChunkId chunkId(startChunk + j * rows + i);
        prefetchManager->recordAccess(chunkId, clientId, 0, 1024);
      }
    }
  }

  Components components;
  std::unique_ptr<StorageOperator::Config> storageOperatorConfig;
  std::unique_ptr<MockStorageOperator> mockStorageOperator;
  std::unique_ptr<PrefetchManager> prefetchManager;
};

TEST_F(PrefetchManagerTest, DetectsSequentialPattern) {
  uint64_t clientId = 42;
  uint64_t startChunk = 1000;
  
  // Record sequential access pattern
  recordSequentialAccess(clientId, startChunk, 10);
  
  // Get predictions
  auto predictions = prefetchManager->getPredictedChunks(clientId);
  
  // Expect predictions for next chunks in sequence
  ASSERT_FALSE(predictions.empty());
  EXPECT_EQ(predictions[0].value(), startChunk + 10);
  
  // Verify that ML framework detection doesn't falsely detect for this pattern
  auto framework = prefetchManager->detectMLFramework(clientId);
  EXPECT_EQ(framework.frameworkType, 0);  // Unknown
}

TEST_F(PrefetchManagerTest, DetectsStridedPattern) {
  uint64_t clientId = 43;
  uint64_t startChunk = 2000;
  int stride = 3;
  
  // Record strided access pattern
  recordStridedAccess(clientId, startChunk, stride, 8);
  
  // Get predictions
  auto predictions = prefetchManager->getPredictedChunks(clientId);
  
  // Expect predictions for next chunks with the same stride
  ASSERT_FALSE(predictions.empty());
  EXPECT_EQ(predictions[0].value(), startChunk + 8 * stride);
}

TEST_F(PrefetchManagerTest, DetectsTransposePattern) {
  uint64_t clientId = 44;
  uint64_t startChunk = 3000;
  
  // Record transpose-like access pattern
  recordTransposeAccess(clientId, startChunk, 4, 4);
  
  // Get predictions
  auto predictions = prefetchManager->getPredictedChunks(clientId);
  
  // We should have some predictions for the transpose pattern
  ASSERT_FALSE(predictions.empty());
}

TEST_F(PrefetchManagerTest, TriggersPrefetching) {
  uint64_t clientId = 45;
  uint64_t startChunk = 4000;
  
  // Setup expectations for the mock
  EXPECT_CALL(*mockStorageOperator, prefetchChunk(::testing::_))
      .WillRepeatedly([](const ChunkId& chunkId) -> CoTryTask<bool> {
        co_return true;
      });
  
  // Record sequential access to generate predictions
  recordSequentialAccess(clientId, startChunk, 6);
  
  // Initialize the prefetch manager
  auto result = prefetchManager->init();
  ASSERT_TRUE(result);
  
  // Trigger prefetching manually
  auto prefetchCount = taskSync(prefetchManager->triggerPrefetch(clientId));
  ASSERT_TRUE(prefetchCount);
  
  // We should have prefetched some chunks
  EXPECT_GT(*prefetchCount, 0);
}

TEST_F(PrefetchManagerTest, DetectsMLBatchPattern) {
  uint64_t clientId = 46;
  uint64_t startChunk = 5000;
  
  // Create a pattern that looks like ML batch processing
  // First batch
  for (int i = 0; i < 5; i++) {
    ChunkId chunkId(startChunk + i);
    prefetchManager->recordAccess(chunkId, clientId, 0, 1024);
  }
  
  // Second batch
  for (int i = 0; i < 5; i++) {
    ChunkId chunkId(startChunk + 100 + i);
    prefetchManager->recordAccess(chunkId, clientId, 0, 1024);
  }
  
  // Third batch
  for (int i = 0; i < 5; i++) {
    ChunkId chunkId(startChunk + 200 + i);
    prefetchManager->recordAccess(chunkId, clientId, 0, 1024);
  }
  
  // Get predictions
  auto predictions = prefetchManager->getPredictedChunks(clientId);
  
  // Check ML framework detection
  auto framework = prefetchManager->detectMLFramework(clientId);
  
  // Expect this to be detected as PyTorch
  EXPECT_EQ(framework.frameworkName, "PyTorch");
  EXPECT_GT(framework.confidence, 0.0f);
}

} // namespace hf3fs::storage 