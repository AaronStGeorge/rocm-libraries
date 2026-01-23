/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_frontend.hpp>

#include <tuple>
#include <vector>

namespace hipdnn_integration_tests {

// Forward declaration - defined in IntegrationGraphVerificationHarness.hpp
template <typename TestCase>
struct EngineTestCase;

/// Filters (engine, testCase) combinations based on engine capability.
///
/// For each test case, builds the graph and queries which engines support it
/// using the frontend API's get_ranked_engine_ids().
///
/// Usage:
///   INSTANTIATE_TEST_SUITE_P(Smoke, MyFixture,
///       testing::ValuesIn(FilteredCombine<MyFixture, TestCaseType>(
///           testing::Combine(
///               testing::Values(TensorLayout::NCHW),
///               testing::ValuesIn(getTestCases())))),
///       EngineTestNameGenerator<TestCaseType>);
///
/// Requirements:
///   FixtureClass must provide:
///     static std::pair<graph::Graph, GraphOutputs> buildGraph(
///         hipdnnHandle_t handle, const TestCase& tc);
///
///   where buildGraph calls validate() and build_operation_graph(handle)
///   before returning.
///
template <typename FixtureClass, typename TestCase>
std::vector<EngineTestCase<TestCase>> FilteredCombine(
    testing::internal::ParamGenerator<TestCase> testCaseGen) {

    std::vector<EngineTestCase<TestCase>> result;

    // Create handle for capability queries
    // Plugin loading is cached (singleton), so this is cheap after first call
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    for (auto it = testCaseGen.begin(); it != testCaseGen.end(); ++it) {
        const TestCase& testCase = *it;

        auto [graph, outputs] = FixtureClass::buildGraph(handle, testCase);

        // Query which engines support this graph
        std::vector<int64_t> engineIds;
        auto status = graph.get_ranked_engine_ids(engineIds);
        if(status.is_bad())
        {
            // If no currently loaded engine supports the graph an error is
            // returned.
            // NOTE: this could be masking a more serious error. An error here
            // could mean that something is broken, or that there aren't loaded
            // engines supporting this graph - we assume the latter.
            continue;
        }

        for (int64_t engineId : engineIds) {
            result.push_back(EngineTestCase<TestCase>{engineId, testCase});
        }
    }

    hipdnnDestroy(handle);
    return result;
}

} // namespace hipdnn_integration_tests