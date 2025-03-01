target_add_test(test_monitor_collector_service test-fabric-lib)

set(TESTS
    # Add test files here when needed
)

# Add each test as a separate executable
foreach(test_file ${TESTS})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    
    target_link_libraries(${test_name} PRIVATE
        monitor_collector_service
        gtest
        gtest_main
        gmock
    )
    
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach() 