/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>

#include "common/EngineDiscovery.hpp"
#include "common/TestGraphBuilders.hpp"

#include <iostream>

namespace {

void demonstrateCapabilityQueries()
{
    using namespace hipdnn_integration_tests;

    std::cout << "=== Pre-InitGoogleTest Engine Capability Demo ===" << std::endl;

    // 1. Discover all engines
    auto allEngines = EngineDiscovery::discoverAllEngines();
    std::cout << "Found " << allEngines.size() << " total engine(s)" << std::endl;
    for (const auto& engine : allEngines) {
        std::cout << "  Engine " << engine.engineId << " from: " << engine.pluginPath << std::endl;
    }

    // 2. Build batchnorm graph, query support
    std::cout << "\nBuilding batchnorm inference graph..." << std::endl;
    auto bnGraph = buildBatchnormInferenceGraph();
    auto bnSupport = EngineDiscovery::getEnginesSupportingGraph(bnGraph);
    std::cout << "Engines supporting batchnorm inference: " << bnSupport.size() << std::endl;
    for (const auto& engine : bnSupport) {
        std::cout << "  Engine " << engine.engineId << std::endl;
    }

    // 3. Build conv graph, query support
    std::cout << "\nBuilding convolution fprop graph..." << std::endl;
    auto convGraph = buildConvFpropGraph();
    auto convSupport = EngineDiscovery::getEnginesSupportingGraph(convGraph);
    std::cout << "Engines supporting conv fprop: " << convSupport.size() << std::endl;
    for (const auto& engine : convSupport) {
        std::cout << "  Engine " << engine.engineId << std::endl;
    }

    std::cout << "=== End Capability Demo ===\n" << std::endl;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    // Demonstrate capability queries BEFORE InitGoogleTest
    demonstrateCapabilityQueries();

    // ::testing::InitGoogleTest(&argc, argv);

    // hipdnn_frontend::initializeFrontendLogging();

    // // Register HipErrorHandler to check and clear HIP errors after each test
    // testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    // listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    // auto result = RUN_ALL_TESTS();
    // spdlog::shutdown();
    // return result;
    return 0;
}