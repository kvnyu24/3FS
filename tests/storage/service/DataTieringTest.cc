#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

#include "common/utils/Coroutine.h"
#include "common/utils/Result.h"
#include "storage/service/Components.h"
#include "storage/service/DataTiering.h"
#include "storage/store/Chunk.h"
#include "storage/store/StorageTargets.h"

namespace hf3fs::storage {

class MockStorageStore : public StorageStore {
public:
  MOCK_METHOD(ChunkPtr, getChunk, (const ChunkId &), (override));
  MOCK_METHOD(Result<ChunkPtr>, createChunk, (const ChunkId &, const ChunkOptions &), (override));
  MOCK_METHOD(Result<void>, removeChunk, (const ChunkId &), (override));
};

class DataTieringTest : public ::testing::Test {
protected:
  void SetUp() override {
    mockStore = std::make_shared<MockStorageStore>();
    components.store.reset(mockStore.get());
    
    // Create the configuration for data tiering
    DataTiering::Config config;
    config.enabled() = true;
    config.tiering_check_interval_ms() = 1000;
    config.hot_to_cold_threshold_seconds() = 60;  // Use small values for testing
    config.cold_to_archive_threshold_seconds() = 120;
    
    // Create the data tiering service
    dataTiering = std::make_unique<DataTiering>(config, components);
  }

  void TearDown() override {
    // Stop the data tiering service
    if (dataTiering) {
      dataTiering->stopAndJoin();
    }
    
    // Unset the store pointer before destroying mockStore
    components.store.reset();
  }

  // Helper function to create a mock chunk
  ChunkPtr createMockChunk(const ChunkId& id, uint64_t size) {
    auto chunk = std::make_shared<Chunk>(id);
    ChunkMetadata metadata;
    metadata.size = size;
    chunk->setMetadata(metadata);
    return chunk;
  }

  // Components structure that DataTiering needs
  Components components;
  std::shared_ptr<MockStorageStore> mockStore;
  std::unique_ptr<DataTiering> dataTiering;
};

TEST_F(DataTieringTest, RecordAccess) {
  ChunkId chunkId(123);
  
  // Record read access
  dataTiering->recordAccess(chunkId, false);
  
  // Get the tier - should be HOT by default
  EXPECT_EQ(dataTiering->getTier(chunkId), StorageTier::HOT);
  
  // Verify stats were recorded
  auto it = dataTiering->chunkStats_.find(chunkId);
  ASSERT_NE(it, dataTiering->chunkStats_.end());
  EXPECT_EQ(it->second.readCount, 1);
  EXPECT_EQ(it->second.writeCount, 0);
  
  // Record write access
  dataTiering->recordAccess(chunkId, true);
  
  // Verify stats were updated
  EXPECT_EQ(it->second.readCount, 1);
  EXPECT_EQ(it->second.writeCount, 1);
}

TEST_F(DataTieringTest, CalculateAppropriatesTier) {
  ChunkId chunkId(456);
  ChunkAccessStats stats;
  
  // Test HOT tier for recent access
  stats.lastAccessTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  stats.accessFrequency = 0.2;
  EXPECT_EQ(dataTiering->calculateAppropriatesTier(stats), StorageTier::HOT);
  
  // Test COLD tier for older access
  stats.lastAccessTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count() - 90;  // 90 seconds ago
  stats.accessFrequency = 0.05;
  EXPECT_EQ(dataTiering->calculateAppropriatesTier(stats), StorageTier::COLD);
  
  // Test ARCHIVE tier for very old access
  stats.lastAccessTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count() - 150;  // 150 seconds ago
  stats.accessFrequency = 0.01;
  EXPECT_EQ(dataTiering->calculateAppropriatesTier(stats), StorageTier::ARCHIVE);
  
  // Test that high frequency keeps in HOT tier regardless of last access time
  stats.lastAccessTime = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count() - 150;  // 150 seconds ago
  stats.accessFrequency = 0.2;  // High frequency
  EXPECT_EQ(dataTiering->calculateAppropriatesTier(stats), StorageTier::HOT);
}

TEST_F(DataTieringTest, MoveToTier) {
  ChunkId chunkId(789);
  auto chunk = createMockChunk(chunkId, 2 * 1024 * 1024);  // 2MB
  
  // Set up the mock store to return our chunk
  EXPECT_CALL(*mockStore, getChunk(chunkId))
      .WillRepeatedly(::testing::Return(chunk));
  
  // Initialize data tiering
  auto result = dataTiering->init();
  ASSERT_TRUE(result);
  
  // Record some access
  dataTiering->recordAccess(chunkId, false);
  
  // Move the chunk to COLD tier
  auto moveResult = taskSync(dataTiering->moveToTier(chunkId, StorageTier::COLD));
  ASSERT_TRUE(moveResult);
  EXPECT_TRUE(*moveResult);
  
  // Verify tier was updated
  EXPECT_EQ(dataTiering->getTier(chunkId), StorageTier::COLD);
  
  // Move back to HOT tier
  moveResult = taskSync(dataTiering->moveToTier(chunkId, StorageTier::HOT));
  ASSERT_TRUE(moveResult);
  EXPECT_TRUE(*moveResult);
  
  // Verify tier was updated
  EXPECT_EQ(dataTiering->getTier(chunkId), StorageTier::HOT);
}

TEST_F(DataTieringTest, CheckAndMoveTier) {
  ChunkId chunkId(101112);
  auto chunk = createMockChunk(chunkId, 5 * 1024 * 1024);  // 5MB
  
  // Set up the mock store to return our chunk
  EXPECT_CALL(*mockStore, getChunk(chunkId))
      .WillRepeatedly(::testing::Return(chunk));
  
  // Initialize data tiering
  auto result = dataTiering->init();
  ASSERT_TRUE(result);
  
  // Record access from a long time ago
  dataTiering->recordAccess(chunkId, false);
  
  // Manually update the last access time to simulate old data
  auto now = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  dataTiering->chunkStats_.upsert(
      chunkId,
      [now](ChunkAccessStats& stats) {
        stats.lastAccessTime = now - 90;  // 90 seconds ago
      });
  
  // Check and move tier
  auto moveResult = taskSync(dataTiering->checkAndMoveTier(chunkId));
  ASSERT_TRUE(moveResult);
  EXPECT_TRUE(*moveResult);
  
  // Verify tier was updated to COLD due to old access time
  EXPECT_EQ(dataTiering->getTier(chunkId), StorageTier::COLD);
}

} // namespace hf3fs::storage 