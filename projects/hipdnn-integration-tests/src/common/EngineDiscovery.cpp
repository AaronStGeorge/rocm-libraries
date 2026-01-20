/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include "EngineDiscovery.hpp"

#include <dlfcn.h>
#include <cstdlib>
#include <iostream>

namespace hipdnn_integration_tests {

// ============================================================================
// DlopenDeleter implementation
// ============================================================================

void DlopenDeleter::operator()(pointer p) const noexcept
{
    if (p) dlclose(p);
}

// ============================================================================
// PluginHandle implementation
// ============================================================================

PluginHandle::PluginHandle(const std::string& pluginPath)
    : pluginPath_(pluginPath)
    , dlopenHandle_(dlopen(pluginPath.c_str(), RTLD_NOW | RTLD_LOCAL))
{
    if (!dlopenHandle_) {
        std::cerr << "Failed to load plugin " << pluginPath << ": " << dlerror() << std::endl;
        return;
    }

    // Load function pointers
    getAllEngineIdsFn_ = reinterpret_cast<GetAllEngineIdsFn>(
        dlsym(dlopenHandle_.get(), "hipdnnEnginePluginGetAllEngineIds"));
    getApplicableEngineIdsFn_ = reinterpret_cast<GetApplicableEngineIdsFn>(
        dlsym(dlopenHandle_.get(), "hipdnnEnginePluginGetApplicableEngineIds"));

    using CreateFn = hipdnnPluginStatus_t (*)(hipdnnEnginePluginHandle_t*);
    auto createFn = reinterpret_cast<CreateFn>(
        dlsym(dlopenHandle_.get(), "hipdnnEnginePluginCreate"));
    auto destroyFn = reinterpret_cast<PluginHandleDeleter::DestroyFn>(
        dlsym(dlopenHandle_.get(), "hipdnnEnginePluginDestroy"));

    if (!getAllEngineIdsFn_) {
        std::cerr << "Plugin missing hipdnnEnginePluginGetAllEngineIds" << std::endl;
        dlopenHandle_.reset();
        return;
    }

    // Create plugin handle for capability queries if APIs are available
    if (createFn && destroyFn && getApplicableEngineIdsFn_) {
        hipdnnEnginePluginHandle_t rawHandle = nullptr;
        if (createFn(&rawHandle) == HIPDNN_PLUGIN_STATUS_SUCCESS && rawHandle) {
            pluginHandle_ = PluginHandlePtr(rawHandle, PluginHandleDeleter{destroyFn});
        } else {
            std::cerr << "Failed to create plugin handle for " << pluginPath << std::endl;
        }
    }
}

bool PluginHandle::isValid() const
{
    return dlopenHandle_ && getAllEngineIdsFn_;
}

std::vector<int64_t> PluginHandle::getAllEngineIds() const
{
    if (!getAllEngineIdsFn_) {
        return {};
    }

    // Two-call pattern: first get count
    uint32_t numEngines = 0;
    if (getAllEngineIdsFn_(nullptr, 0, &numEngines) != HIPDNN_PLUGIN_STATUS_SUCCESS) {
        return {};
    }

    if (numEngines == 0) {
        return {};
    }

    // Second call: get IDs
    std::vector<int64_t> engineIds(numEngines);
    if (getAllEngineIdsFn_(engineIds.data(), numEngines, &numEngines) != HIPDNN_PLUGIN_STATUS_SUCCESS) {
        return {};
    }

    return engineIds;
}

std::vector<int64_t> PluginHandle::getApplicableEngineIds(
    const flatbuffers::DetachedBuffer& graphBuffer) const
{
    if (!pluginHandle_ || !getApplicableEngineIdsFn_) {
        return {};
    }

    // Wrap buffer in hipdnnPluginConstData_t
    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = graphBuffer.data();
    opGraph.size = graphBuffer.size();

    // Use getAllEngineIds() count as maxEngines (matches backend pattern)
    auto allEngines = getAllEngineIds();
    if (allEngines.empty()) {
        return {};
    }

    const auto maxEngines = static_cast<uint32_t>(allEngines.size());
    std::vector<int64_t> engineIds(maxEngines);
    uint32_t numEngines = 0;

    if (getApplicableEngineIdsFn_(pluginHandle_.get(), &opGraph, engineIds.data(), maxEngines, &numEngines)
        != HIPDNN_PLUGIN_STATUS_SUCCESS) {
        return {};
    }

    engineIds.resize(numEngines);
    return engineIds;
}

// ============================================================================
// EngineDiscovery implementation
// ============================================================================

std::vector<std::string> EngineDiscovery::discoverPluginPaths()
{
    std::vector<std::string> paths;

    const char* envPath = std::getenv("HIPDNN_TEST_PLUGIN_PATH");
    if (envPath && envPath[0] != '\0') {
        // For now, treat as a single path. Could extend to colon-separated list.
        paths.emplace_back(envPath);
    }

    return paths;
}

std::vector<EngineInfo> EngineDiscovery::discoverAllEngines()
{
    std::vector<EngineInfo> allEngines;

    for (const auto& pluginPath : discoverPluginPaths()) {
        PluginHandle plugin(pluginPath);
        if (!plugin.isValid()) {
            continue;
        }

        for (int64_t engineId : plugin.getAllEngineIds()) {
            allEngines.push_back({engineId, pluginPath});
        }
    }

    return allEngines;
}

std::vector<EngineInfo> EngineDiscovery::getEnginesSupportingGraph(
    const flatbuffers::DetachedBuffer& graphBuffer)
{
    std::vector<EngineInfo> supportingEngines;

    for (const auto& pluginPath : discoverPluginPaths()) {
        PluginHandle plugin(pluginPath);
        if (!plugin.isValid()) {
            continue;
        }

        for (int64_t engineId : plugin.getApplicableEngineIds(graphBuffer)) {
            supportingEngines.push_back({engineId, pluginPath});
        }
    }

    return supportingEngines;
}

} // namespace hipdnn_integration_tests