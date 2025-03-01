# ML Workload Analytics Service

## Overview

The ML Workload Analytics service is designed to monitor, analyze, and report on machine learning workload patterns in the 3FS (Fire-Flyer File System). It captures file access patterns, infers the type of ML workload, detects the ML framework being used, and provides insights to optimize storage performance.

## Key Components

### Data Structures

- **WorkloadStats**: Stores statistics about ML workloads, including operation counts, access patterns, and performance metrics.
- **OperationRecord**: Represents a single file system operation with metadata such as type, client ID, path, timestamp, and latency.

### Enums

- **WorkloadType**: Classifies workloads as TRAINING, INFERENCE, DATA_PREP, EVALUATION, CHECKPOINT, or UNKNOWN.
- **FrameworkType**: Identifies the ML framework as PYTORCH, TENSORFLOW, JAX, MXNET, CUSTOM, or UNKNOWN.
- **OperationType**: Categorizes file system operations as READ, WRITE, OPEN, CLOSE, STAT, LIST, CREATE, MKDIR, REMOVE, or RENAME.

### Main Class: MLWorkloadAnalytics

The `MLWorkloadAnalytics` class is the core of the service, providing the following functionality:

1. **Configuration**: Configurable parameters for analysis intervals, reporting frequency, and detection thresholds.
2. **Recording Operations**: Captures file system operations and updates workload statistics.
3. **Pattern Detection**: Analyzes access patterns to identify sequential, random, batch, or checkpoint patterns.
4. **Workload Classification**: Determines the type of ML workload based on access patterns and file types.
5. **Framework Detection**: Identifies the ML framework based on file extensions and path patterns.
6. **Reporting**: Generates reports on workload characteristics and performance metrics.
7. **Historical Analysis**: Maintains historical records of workload activity for trend analysis.

## Integration Points

The ML Workload Analytics service integrates with the 3FS storage layer through the `StorageOperator` class. Key integration points include:

1. **Operation Recording**: The `StorageOperator` passes file system operations to the analytics service through the `recordOperation` method.
2. **Management API**: Exposed through the `StorageMgmt` class to provide analytics data to management tools.
3. **Prefetching Integration**: Provides insights to the `PrefetchManager` to optimize data access.

## Usage Example

```cpp
// Initialize the service
MLWorkloadAnalytics::Config config;
config.set_enabled(true);
auto analytics = std::make_unique<MLWorkloadAnalytics>(config);
analytics->init();

// Record a file operation
OperationRecord record;
record.type = OperationType::READ;
record.clientId = clientId;
record.path = "/datasets/imagenet/batch_1.pt";
record.timestamp = getCurrentTimeMs();
record.sizeBytes = 10 * 1024 * 1024;  // 10MB
record.latencyUs = 15000;             // 15ms
record.offset = 0;
record.jobId = "training-job-123";

analytics->recordOperation(record);

// Get workload stats
auto stats = analytics->getWorkloadStats(clientId);
if (stats) {
  std::cout << "Workload type: " << static_cast<int>(stats->type) << std::endl;
  std::cout << "Framework: " << static_cast<int>(stats->framework) << std::endl;
  std::cout << "Access pattern: " << stats->detectedWorkloadPattern << std::endl;
}

// Generate a report
std::string report = analytics->getWorkloadReport();
```

## Testing

The service includes comprehensive tests in `standalone_test.cc` that validate:

1. Sequential access pattern detection
2. Workload type detection
3. ML framework detection

To run the tests:

```bash
make -f Makefile.simple run_standalone
```

## Implementation Notes

- The service uses background threads for periodic analysis and reporting.
- Access patterns are detected using statistical analysis of access history.
- Framework detection uses heuristics based on file extensions and path patterns.
- Workload classification combines multiple signals including read/write ratios, file types, and access patterns.
- Historical data is retained for a configurable period (default: 24 hours).

## Future Enhancements

1. Deep learning-based pattern recognition for more accurate workload classification
2. Integration with ClickHouse for long-term analytics storage
3. Real-time anomaly detection in ML workload patterns
4. Adaptive tiering based on workload analysis
5. Predictive I/O optimization based on historical patterns 