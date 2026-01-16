cmake_minimum_required(VERSION 3.25.2)
project(hipdnn-integration-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)
find_package(hipdnn_data_sdk REQUIRED)
find_package(hipdnn_backend REQUIRED)
find_package(hipdnn_test_sdk REQUIRED)
find_package(hipdnn_plugin_sdk REQUIRED)
find_package(hipdnn_frontend REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

add_executable(hipdnn_integration_tests
    src/main.cpp
    src/list_plugins_test.cpp
    src/IntegrationGpuConvForward.cpp
    src/IntegrationGpuConvBackwardData.cpp
    src/IntegrationGpuConvBackwardWeights.cpp
    src/IntegrationGpuConvFwdBiasActiv.cpp
    src/IntegrationGpuBatchnormForwardInference.cpp
    src/IntegrationGpuBatchnormForwardTraining.cpp
    src/IntegrationGpuBatchnormFwdPlusActive.cpp
    src/IntegrationGpuBatchnormFwdTrainingActiv.cpp
    src/IntegrationGpuBatchnormBackward.cpp
    src/IntegrationGpuBatchnormBackwardActivation.cpp
)

target_include_directories(hipdnn_integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(hipdnn_integration_tests
    PRIVATE
        hip::host
        hipdnn_data_sdk
        hipdnn_backend
        hipdnn_test_sdk
        hipdnn_plugin_sdk
        hipdnn_frontend
        GTest::gtest
        Threads::Threads
)

# No PLUGIN_PATH compile definition - uses HIPDNN_TEST_PLUGIN_PATH env var
# No add_dependencies on specific plugin - tests are plugin-agnostic

include(GNUInstallDirs)
install(TARGETS hipdnn_integration_tests
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Generate CTestTestfile for installed tests (used by CI test runners).
# We manually create this file with relative paths - CMake's CTest module
# generates files with absolute paths which don't work for relocated installs.
set(_INSTALLED_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")
file(WRITE "${_INSTALLED_CTEST_FILE}"
    "# CTestTestfile for installed hipdnn-integration-tests\n"
    "# Used by CI test runners to execute tests from the installed location\n\n"
    "add_test(hipdnn_integration_tests \"../hipdnn_integration_tests\")\n"
)

install(
    FILES "${_INSTALLED_CTEST_FILE}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/hipdnn_integration_tests_test_infra"
    RENAME CTestTestfile.cmake
)
