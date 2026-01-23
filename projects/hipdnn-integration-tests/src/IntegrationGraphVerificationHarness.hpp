// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMiopenRmsValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>

namespace hipdnn_integration_tests
{

// Test parameter that pairs an engine ID with a test case.
// Used by FilteredCombine to return (engine, testCase) pairs where
// the engine has been verified to support the test case's graph.
template <typename TestCase>
struct EngineTestCase
{
    int64_t engineId;
    TestCase testCase;
};

// Name generator for EngineTestCase-parameterized tests.
// Produces names like "Fusilli_0", "MIOpen_1", etc.
// Falls back to numeric ID if engine name isn't registered.
template <typename TestCase>
std::string EngineTestNameGenerator(const testing::TestParamInfo<EngineTestCase<TestCase>>& info)
{
    std::string engineName;
    // TODO: add fusilli plugin to rocm-libraries/projects/hipdnn/data_sdk/tests/utilities/TestEngineNames.cpp
    try
    {
        engineName = std::string(hipdnn_data_sdk::utilities::getEngineNameFromId(info.param.engineId));
    }
    catch(const std::out_of_range&)
    {
        engineName = "Engine" + std::to_string(info.param.engineId);
    }
    return engineName + "_" + std::to_string(info.index);
}

// NOLINTBEGIN (portability-template-virtual-member-function)
template <typename DataType, typename TestCaseType>
class IntegrationGraphVerificationHarness : public ::testing::TestWithParam<TestCaseType>
{
protected:
    hipdnnHandle_t _handle = nullptr;
    hipStream_t _stream = nullptr;
    int _deviceId = 0;
    std::unordered_map<int64_t, std::string> _tensorIdToNameMap;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_test_sdk::utilities::IReferenceValidation>>
        _tensorIdToValidatorMap;
    std::vector<std::function<void()>> _deferredValidators;

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        // Initialize HIP
        ASSERT_EQ(hipInit(0), hipSuccess);
        ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

        // Create handle and stream (hipDNN auto-loads plugins from standard path)
        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
        ASSERT_EQ(hipStreamCreate(&_stream), hipSuccess);
        ASSERT_EQ(hipdnnSetStream(_handle, _stream), HIPDNN_STATUS_SUCCESS);

        // Verify loaded plugins exactly match expected plugins
        verifyExpectedPlugins();
    }

    // TODO: Extract config parsing into a dedicated TestConfig class that can be
    // hydrated from JSON and reused across test infrastructure.
    void verifyExpectedPlugins()
    {
        const char* configPathEnv = std::getenv("HIPDNN_TEST_CONFIG_PATH");
        if(configPathEnv == nullptr || std::strlen(configPathEnv) == 0)
        {
            FAIL() << "HIPDNN_TEST_CONFIG_PATH environment variable not set";
        }

        std::filesystem::path configPath = std::filesystem::weakly_canonical(configPathEnv);

        // Parse JSON config
        std::ifstream configFile(configPath);
        if(!configFile.is_open())
        {
            FAIL() << "Failed to open config file: " << configPath;
        }

        nlohmann::json config;
        try
        {
            config = nlohmann::json::parse(configFile);
        }
        catch(const nlohmann::json::parse_error& e)
        {
            FAIL() << "Failed to parse config JSON: " << e.what();
        }

        // Extract expected plugin filenames
        std::set<std::string> expectedPlugins;
        if(config.contains("plugins"))
        {
            for(const auto& [name, info] : config["plugins"].items())
            {
                if(info.contains("path"))
                {
                    // Config contains filenames only (e.g., "libfusilli_plugin.so")
                    expectedPlugins.insert(info["path"].get<std::string>());
                }
            }
        }

        // Get actual loaded plugin filenames
        auto loadedPlugins = getLoadedPluginFilenames();

        // Verify exact match
        if(expectedPlugins != loadedPlugins)
        {
            FAIL() << "Plugin mismatch!\n"
                   << "  Expected: " << formatPluginSet(expectedPlugins) << "\n"
                   << "  Loaded:   " << formatPluginSet(loadedPlugins);
        }
    }

    std::set<std::string> getLoadedPluginFilenames()
    {
        size_t numPlugins    = 0;
        size_t maxPathLength = 0;
        auto   status
            = hipdnnGetLoadedEnginePluginPaths_ext(_handle, &numPlugins, nullptr, &maxPathLength);

        if(status != HIPDNN_STATUS_SUCCESS || numPlugins == 0)
        {
            return {};
        }

        std::vector<std::vector<char>> pathBuffers(numPlugins, std::vector<char>(maxPathLength));
        std::vector<char*>             pluginPathsC(numPlugins);
        for(size_t i = 0; i < numPlugins; ++i)
        {
            pluginPathsC[i] = pathBuffers[i].data();
        }

        status = hipdnnGetLoadedEnginePluginPaths_ext(
            _handle, &numPlugins, pluginPathsC.data(), &maxPathLength);
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            return {};
        }

        std::set<std::string> filenames;
        for(size_t i = 0; i < numPlugins; ++i)
        {
            std::filesystem::path pluginPath =
                std::filesystem::canonical(pluginPathsC[i]);
            filenames.insert(pluginPath.filename().string());
        }
        return filenames;
    }

    static std::string formatPluginSet(const std::set<std::string>& plugins)
    {
        std::string result = "[";
        bool        first  = true;
        for(const auto& p : plugins)
        {
            if(!first)
                result += ", ";
            result += p;
            first = false;
        }
        result += "]";
        return result;
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
        }
        if(_stream != nullptr)
        {
            ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
        }
    }

    virtual void runGraphTest(DataType tolerance) = 0;

    void verifyGraph(hipdnn_frontend::graph::Graph& graph, unsigned int seed)
    {
        hipdnn_test_sdk::utilities::GraphTensorBundle gpuBundle, cpuBundle;
        std::vector<int64_t> outputTensorIds;

        auto result = graph.build(_handle);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;

        generateBundles(graph, cpuBundle, gpuBundle, outputTensorIds);

        initializeBundle(graph, gpuBundle, seed);
        initializeBundle(graph, cpuBundle, seed);

        ASSERT_NO_FATAL_FAILURE(executeGpuGraph(_handle, graph, gpuBundle));
        executeCpuGraph(graph, cpuBundle);

        ASSERT_GE(outputTensorIds.size(), 1)
            << "At least one output tensor id must be specified for "
               "validation.";

        HIPDNN_LOG_INFO("Validating {} output tensors", outputTensorIds);

        // Lazily register validators after graph execution since tensor Ids and types may be inferred during graph finalization
        for(const auto& registerValidator : _deferredValidators)
        {
            registerValidator();
        }

        for(const auto& tensorId : outputTensorIds)
        {
            auto& cpuTensor = cpuBundle.tensors.at(tensorId);
            auto& gpuTensor = gpuBundle.tensors.at(tensorId);

            // This tells the tensor that its data has been modified on the device side
            // All frontend graph knows is a (void*) pointer to device memory, so we need to inform the tensor
            // that the data there is now valid so that it knows to copy from device to host when requested
            // by the validation step.
            gpuTensor->markDeviceModified();

            if(_tensorIdToValidatorMap.find(tensorId) == _tensorIdToValidatorMap.end())
            {
                FAIL() << "No validator registered for tensor with id: " << tensorId
                       << ", name: " << getOutputTensorName(tensorId);
            }

            bool valid = _tensorIdToValidatorMap.at(tensorId)->allClose(*cpuTensor, *gpuTensor);
            ASSERT_TRUE(valid) << "Mismatch found in tensor with id: " << tensorId
                               << ", name: " << _tensorIdToNameMap.at(tensorId);
        }
    }

    void registerValidator(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr,
                           float tolerance)
    {
        registerValidator(attr, tolerance, tolerance);
    }

    void registerValidator(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr,
                           float absoluteTolerance,
                           float relativeTolerance)
    {
        // Since the graph can infer properties + Ids, we defer validator registration until right before validation in verifyGraph
        _deferredValidators.emplace_back([=]() {
            _tensorIdToValidatorMap.insert(
                {attr->get_uid(),
                 hipdnn_test_sdk::utilities::createAllCloseValidator(
                     toSdkType(attr->get_data_type()), absoluteTolerance, relativeTolerance)});
            _tensorIdToNameMap.insert({attr->get_uid(), attr->get_name()});
        });
    }

    void registerRmsValidator(const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> attr,
                              float rmsThreshold)
    {
        // Since the graph can infer properties + Ids, we defer validator registration until right before validation in verifyGraph
        _deferredValidators.emplace_back([=]() {
            _tensorIdToValidatorMap.insert({attr->get_uid(),
                                            hipdnn_test_sdk::utilities::createRmsValidator(
                                                toSdkType(attr->get_data_type()), rmsThreshold)});
            _tensorIdToNameMap.insert({attr->get_uid(), attr->get_name()});
        });
    }

    virtual void generateBundles(hipdnn_frontend::graph::Graph& graph,
                                 hipdnn_test_sdk::utilities::GraphTensorBundle& cpuBundle,
                                 hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle,
                                 std::vector<int64_t>& outputTensorIds)
    {
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle))
                {
                    outputTensorIds.push_back(tensorAttr->get_uid());
                }
            }
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                tryAddTensorToBundles(tensorAttr, cpuBundle, gpuBundle);
            }
        });
    }

    virtual void initializeBundle([[maybe_unused]] const hipdnn_frontend::graph::Graph& graph,
                                  hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                                  unsigned int seed)
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, -1.0f, 1.0f, seed);
        }
    }

private:
    void executeGpuGraph(hipdnnHandle_t handle,
                         hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        int64_t workspaceSize;
        auto result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0) << result.err_msg;
        hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        auto variantPack = bundle.toDeviceVariantPack();
        result = graph.execute(handle, variantPack, workspace.get());
        ASSERT_EQ(result.code, hipdnn_frontend::ErrorCode::OK) << result.err_msg;
    }

    void executeCpuGraph(hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
    {
        auto flatbufferGraph = graph.buildFlatbufferOperationGraph();

        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            flatbufferGraph.data(), flatbufferGraph.size(), bundle.toHostVariantPack());
    }

    std::string getOutputTensorName(int64_t tensorId)
    {
        return _tensorIdToNameMap.at(tensorId);
    }

    bool tryAddTensorToBundles(
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& tensorAttr,
        hipdnn_test_sdk::utilities::GraphTensorBundle& cpuBundle,
        hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle)
    {
        int64_t tensorId = tensorAttr->get_uid();

        if(tensorAttr->get_is_virtual()
           || cpuBundle.tensors.find(tensorId) != cpuBundle.tensors.end())
        {
            return false;
        }

        cpuBundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
        gpuBundle.tensors.insert({tensorId, createTensorFromAttribute(*tensorAttr)});
        _tensorIdToNameMap.insert({tensorId, tensorAttr->get_name()});

        return true;
    }

};

// NOLINTEND (portability-template-virtual-member-function)

} // namespace hipdnn_integration_tests
