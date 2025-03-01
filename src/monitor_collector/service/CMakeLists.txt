set(MONITOR_COLLECTOR_SERVICE_SRCS
    MLWorkloadAnalytics.cc
)

add_library(monitor_collector_service OBJECT ${MONITOR_COLLECTOR_SERVICE_SRCS})

target_include_directories(monitor_collector_service PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(monitor_collector_service PUBLIC
    common_utils
    folly
)

add_dependencies(monitor_collector_service
    common_utils
    folly
) 