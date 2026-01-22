/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <hipdnn_frontend.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    hipdnn_frontend::initializeFrontendLogging();

    // TODO: Re-enable after fixing hipErrorPeerAccessAlreadyEnabled issue from IREE
    // testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    // listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

    auto result = RUN_ALL_TESTS();
    spdlog::shutdown();
    return result;
}