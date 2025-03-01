#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <cassert>

// Include MLWorkloadAnalytics implementation directly
#include "src/monitor_collector/service/MLWorkloadAnalytics.h"

using namespace hf3fs::monitor_collector;

// Simple test function
void test_ml_workload_analytics() {
    // Create config
    MLWorkloadAnalytics::Config config;
    config.set_enabled(true);
    config.set_analysis_interval_ms(100);   // Fast analysis for testing
    config.set_report_interval_ms(500);     // Fast reporting for testing
    config.set_min_ops_for_detection(5);    // Low threshold for testing
    
    // Create analytics service
    auto analytics = std::make_unique<MLWorkloadAnalytics>(config);
    
    // Initialize
    auto result = analytics->init();
    assert(result && "Failed to initialize MLWorkloadAnalytics");
    
    std::cout << "MLWorkloadAnalytics initialized successfully" << std::endl;
    
    // Simulate a training workload
    uint64_t clientId = 1001;
    
    std::cout << "Simulating training workload..." << std::endl;
    
    // First, simulate dataset loading
    for (int i = 0; i < 10; i++) {
        OperationRecord record;
        record.type = OperationType::READ;
        record.clientId = clientId;
        record.path = "/datasets/imagenet/batch_" + std::to_string(i) + ".pt";
        record.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
        record.sizeBytes = 10 * 1024 * 1024; // 10MB
        record.latencyUs = 15000;            // 15ms
        record.offset = i * 10 * 1024 * 1024;
        record.jobId = "training-job-123";
        
        analytics->recordOperation(record);
    }
    
    // Now simulate checkpoint writes
    for (int epoch = 0; epoch < 3; epoch++) {
        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        OperationRecord record;
        record.type = OperationType::WRITE;
        record.clientId = clientId;
        record.path = "/models/checkpoint_" + std::to_string(epoch) + ".pt";
        record.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
        record.sizeBytes = 100 * 1024 * 1024; // 100MB
        record.latencyUs = 100000;            // 100ms
        record.offset = 0;
        record.jobId = "training-job-123";
        
        analytics->recordOperation(record);
    }
    
    // Wait for analysis to run
    std::cout << "Waiting for analysis to run..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    
    // Get stats for the client
    auto stats = analytics->getWorkloadStats(clientId);
    
    if (!stats) {
        std::cerr << "Error: No stats found for client " << clientId << std::endl;
    } else {
        std::cout << "Client stats found:" << std::endl;
        std::cout << "  Workload type: " << static_cast<int>(stats->type) << std::endl;
        std::cout << "  Framework: " << static_cast<int>(stats->framework) << std::endl;
        std::cout << "  Read ops: " << stats->readOps << std::endl;
        std::cout << "  Write ops: " << stats->writeOps << std::endl;
        std::cout << "  Total bytes read: " << stats->totalBytesRead << std::endl;
        std::cout << "  Total bytes written: " << stats->totalBytesWritten << std::endl;
        
        if (!stats->mostAccessedFiles.empty()) {
            std::cout << "  Most accessed file: " << stats->mostAccessedFiles.back() << std::endl;
        }
        
        if (!stats->detectedWorkloadPattern.empty()) {
            std::cout << "  Detected pattern: " << stats->detectedWorkloadPattern << std::endl;
        }
    }
    
    // Get workload report
    std::string report = analytics->getWorkloadReport();
    std::cout << "Workload report: " << report << std::endl;
    
    // Cleanup
    analytics->stopAndJoin();
    
    std::cout << "Test completed successfully" << std::endl;
}

// Main function
int main() {
    try {
        test_ml_workload_analytics();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
} 