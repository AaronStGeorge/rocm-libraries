/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

#include "EngineDiscovery.hpp"

namespace hipdnn_integration_tests {

/// Filters (engine, innerParam) combinations based on engine capability.
///
/// Usage:
///   INSTANTIATE_TEST_SUITE_P(Smoke, MyFixture,
///       testing::ValuesIn(FilteredCombine<MyFixture>(
///           EngineDiscovery::discoverAllEngines(),
///           testing::Combine(
///               testing::Values(TensorLayout::NCHW),
///               testing::ValuesIn(getTestCases())))));
///
/// Requirements:
///   FixtureClass must provide:
///     static flatbuffers::DetachedBuffer buildGraphForTestCase(const InnerParam& tc);
///
template <typename FixtureClass, typename InnerParam>
std::vector<std::tuple<int64_t, InnerParam>> FilteredCombine(
    const std::vector<EngineInfo>& engines,
    testing::internal::ParamGenerator<InnerParam> innerGen) {

    std::vector<std::tuple<int64_t, InnerParam>> result;

    for (const auto& engine : engines) {
        PluginHandle plugin(engine.pluginPath);
        auto allEngineIds = plugin.getAllEngineIds();

        for (auto it = innerGen.begin(); it != innerGen.end(); ++it) {
            const InnerParam& inner = *it;

            // Build graph for this test case and check if engine supports it
            auto graph = FixtureClass::buildGraphForTestCase(inner);
            auto applicable = plugin.getApplicableEngineIds(graph);

            bool supported = std::find(applicable.begin(), applicable.end(),
                                       engine.engineId) != applicable.end();
            if (supported) {
                result.emplace_back(engine.engineId, inner);
            }
        }
    }

    return result;
}

} // namespace hipdnn_integration_tests
