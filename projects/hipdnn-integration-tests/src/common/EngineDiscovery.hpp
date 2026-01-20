/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipdnn_integration_tests {

// ============================================================================
// Custom deleters for RAII handle management
// ============================================================================

/// Deleter for dlopen handles
struct DlopenDeleter {
    using pointer = void*;
    void operator()(pointer p) const noexcept;
};
using DlopenHandle = std::unique_ptr<void, DlopenDeleter>;

/// Stateful deleter for plugin handles (needs destroy function pointer)
struct PluginHandleDeleter {
    using DestroyFn = hipdnnPluginStatus_t (*)(hipdnnEnginePluginHandle_t);
    DestroyFn destroyFn = nullptr;

    void operator()(HipdnnEnginePluginHandle* p) const noexcept {
        if (p && destroyFn) destroyFn(p);
    }
};
using PluginHandlePtr = std::unique_ptr<HipdnnEnginePluginHandle, PluginHandleDeleter>;

// ============================================================================
// Main types
// ============================================================================

struct EngineInfo {
    int64_t engineId;
    std::string pluginPath;
};

/// RAII wrapper for a loaded engine plugin.
/// Handles dlopen/dlclose and provides access to plugin APIs.
class PluginHandle {
public:
    explicit PluginHandle(const std::string& pluginPath);
    ~PluginHandle() = default;

     // Non-copyable, movable
    PluginHandle(PluginHandle&&) = default;
    PluginHandle& operator=(PluginHandle&&) = default;
    PluginHandle(const PluginHandle&) = delete;
    PluginHandle& operator=(const PluginHandle&) = delete;

    bool isValid() const;
    const std::string& getPath() const { return pluginPath_; }

    /// Get all engine IDs this plugin provides
    std::vector<int64_t> getAllEngineIds() const;

    /// Query which engines support a given graph
    std::vector<int64_t> getApplicableEngineIds(
        const flatbuffers::DetachedBuffer& graphBuffer) const;

private:
    std::string pluginPath_;
    DlopenHandle dlopenHandle_;
    PluginHandlePtr pluginHandle_;

    // Function pointers for plugin APIs
    using GetAllEngineIdsFn = hipdnnPluginStatus_t (*)(int64_t*, uint32_t, uint32_t*);
    using GetApplicableEngineIdsFn = hipdnnPluginStatus_t (*)(
        hipdnnEnginePluginHandle_t, const hipdnnPluginConstData_t*, int64_t*, uint32_t, uint32_t*);

    GetAllEngineIdsFn getAllEngineIdsFn_ = nullptr;
    GetApplicableEngineIdsFn getApplicableEngineIdsFn_ = nullptr;
};

/// Static helper class for discovering engines across all plugins
class EngineDiscovery {
public:
    /// Discover plugin paths from HIPDNN_TEST_PLUGIN_PATH environment variable
    static std::vector<std::string> discoverPluginPaths();

    /// Discover all engines across all discoverable plugins
    static std::vector<EngineInfo> discoverAllEngines();

    /// Find engines that support a given operation graph
    static std::vector<EngineInfo> getEnginesSupportingGraph(
        const flatbuffers::DetachedBuffer& graphBuffer);
};

} // namespace hipdnn_integration_tests