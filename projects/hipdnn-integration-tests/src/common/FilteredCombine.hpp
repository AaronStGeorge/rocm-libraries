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

/// Filters (engine, innerParam) combinations based on engine capability.
///
/// For each test case, builds the graph and queries which engines support it
/// using the frontend API's get_ranked_engine_ids().
///
/// Usage:
///   INSTANTIATE_TEST_SUITE_P(Smoke, MyFixture,
///       testing::ValuesIn(FilteredCombine<MyFixture>(
///           testing::Combine(
///               testing::Values(TensorLayout::NCHW),
///               testing::ValuesIn(getTestCases())))));
///
/// Requirements:
///   FixtureClass must provide:
///     static std::pair<graph::Graph, GraphOutputs> buildGraph(
///         hipdnnHandle_t handle, const InnerParam& tc);
///
///   where buildGraph calls validate() and build_operation_graph(handle)
///   before returning.
///
template <typename FixtureClass, typename InnerParam>
std::vector<std::tuple<int64_t, InnerParam>> FilteredCombine(
    testing::internal::ParamGenerator<InnerParam> innerGen) {

    std::vector<std::tuple<int64_t, InnerParam>> result;

    // Create handle for capability queries
    // Plugin loading is cached (singleton), so this is cheap after first call
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    for (auto it = innerGen.begin(); it != innerGen.end(); ++it) {
        const InnerParam& inner = *it;

        auto [graph, outputs] = FixtureClass::buildGraph(handle, inner);

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
            result.emplace_back(engineId, inner);
        }
    }

    hipdnnDestroy(handle);
    return result;
}

} // namespace hipdnn_integration_tests