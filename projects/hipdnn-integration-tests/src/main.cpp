/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <dlfcn.h>

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>

#include <iostream>
#include <vector>
#include <string>

namespace {

// Type alias for the plugin's exported function
using GetAllEngineIdsFn = hipdnnPluginStatus_t (*)(int64_t*, uint32_t, uint32_t*);

std::vector<int64_t> getEngineIdsFromPlugin(const std::string& pluginPath)
{
    void* handle = dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(!handle)
    {
        std::cerr << "Failed to load plugin: " << dlerror() << std::endl;
        return {};
    }

    auto getAllEngineIds
        = reinterpret_cast<GetAllEngineIdsFn>(dlsym(handle, "hipdnnEnginePluginGetAllEngineIds"));
    if(!getAllEngineIds)
    {
        std::cerr << "Failed to find hipdnnEnginePluginGetAllEngineIds: " << dlerror() << std::endl;
        dlclose(handle);
        return {};
    }

    // Two-call pattern: first get count
    uint32_t numEngines = 0;
    if(getAllEngineIds(nullptr, 0, &numEngines) != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        dlclose(handle);
        return {};
    }

    if(numEngines == 0)
    {
        dlclose(handle);
        return {};
    }

    // Second call: get IDs
    std::vector<int64_t> engineIds(numEngines);
    if(getAllEngineIds(engineIds.data(), numEngines, &numEngines) != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        dlclose(handle);
        return {};
    }

    dlclose(handle);
    return engineIds;
}

std::vector<std::string> getLoadedPluginPaths(hipdnnHandle_t handle)
{
    size_t numPlugins    = 0;
    size_t maxPathLength = 0;

    auto status
        = hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins, nullptr, &maxPathLength);
    if(status != HIPDNN_STATUS_SUCCESS || numPlugins == 0)
    {
        return {};
    }

    std::vector<std::vector<char>> pathBuffers(numPlugins, std::vector<char>(maxPathLength));
    std::vector<char*> pluginPathsC(numPlugins);
    for(size_t i = 0; i < numPlugins; ++i)
    {
        pluginPathsC[i] = pathBuffers[i].data();
    }

    status
        = hipdnnGetLoadedEnginePluginPaths_ext(handle, &numPlugins, pluginPathsC.data(), &maxPathLength);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        return {};
    }

    std::vector<std::string> paths;
    for(size_t i = 0; i < numPlugins; ++i)
    {
        paths.emplace_back(pluginPathsC[i]);
    }
    return paths;
}

void enumerateAllEngines()
{
    std::cout << "=== Pre-InitGoogleTest Engine Enumeration ===" << std::endl;

    // Create handle to trigger plugin loading
    hipdnnHandle_t handle = nullptr;
    if(hipdnnCreate(&handle) != HIPDNN_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create hipdnn handle" << std::endl;
        return;
    }

    auto pluginPaths = getLoadedPluginPaths(handle);
    std::cout << "Loaded " << pluginPaths.size() << " plugin(s)" << std::endl;

    for(const auto& path : pluginPaths)
    {
        std::cout << "\nPlugin: " << path << std::endl;
        auto engineIds = getEngineIdsFromPlugin(path);
        std::cout << "  Engines: " << engineIds.size() << std::endl;
        for(auto id : engineIds)
        {
            std::cout << "    - " << id << std::endl;
        }
    }

    hipdnnDestroy(handle);
    std::cout << "=== End Engine Enumeration ===\n" << std::endl;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    // Enumerate engines BEFORE InitGoogleTest
    enumerateAllEngines();

    ::testing::InitGoogleTest(&argc, argv);

    hipdnn_frontend::initializeFrontendLogging();

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    auto result = RUN_ALL_TESTS();
    spdlog::shutdown();
    return result;
}