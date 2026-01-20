/*
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_frontend/Graph.hpp>

namespace hipdnn_integration_tests {

/// Build a batchnorm inference graph (fusilli doesn't support this)
/// Returns a serialized flatbuffer that can be passed to plugin capability queries.
inline flatbuffers::DetachedBuffer buildBatchnormInferenceGraph()
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

    Graph graph;
    graph.set_name("BN_Inference_CapabilityTest")
         .set_compute_data_type(DataType::FLOAT)
         .set_intermediate_data_type(DataType::FLOAT)
         .set_io_data_type(DataType::FLOAT);

    // Input tensor: NCHW format, small test size
    std::vector<int64_t> inputDims = {1, 64, 8, 8};
    std::vector<int64_t> inputStrides = {64 * 8 * 8, 8 * 8, 8, 1};
    auto x = std::make_shared<TensorAttributes>(
        makeTensorAttributes("x", inputDims, inputStrides));
    x->set_uid(1);

    // Channel tensors: [1, C, 1, 1]
    std::vector<int64_t> channelDims = {1, 64, 1, 1};
    std::vector<int64_t> channelStrides = {64, 1, 1, 1};

    auto mean = std::make_shared<TensorAttributes>(
        makeTensorAttributes("mean", DataType::FLOAT, channelDims, channelStrides));
    mean->set_uid(2);

    auto invVariance = std::make_shared<TensorAttributes>(
        makeTensorAttributes("inv_variance", DataType::FLOAT, channelDims, channelStrides));
    invVariance->set_uid(3);

    auto scale = std::make_shared<TensorAttributes>(
        makeTensorAttributes("scale", DataType::FLOAT, channelDims, channelStrides));
    scale->set_uid(4);

    auto bias = std::make_shared<TensorAttributes>(
        makeTensorAttributes("bias", DataType::FLOAT, channelDims, channelStrides));
    bias->set_uid(5);

    // Build batchnorm inference operation
    BatchnormInferenceAttributes bnAttrs;
    auto y = graph.batchnorm_inference(x, mean, invVariance, scale, bias, bnAttrs);
    y->set_output(true);

    // Validate and serialize
    auto result = graph.validate();
    if (result.code != ErrorCode::OK) {
        throw std::runtime_error("Failed to validate batchnorm graph: " + result.get_message());
    }

    return graph.buildFlatbufferOperationGraph();
}

/// Build a convolution forward graph (fusilli supports this via composable_kernel)
/// Returns a serialized flatbuffer that can be passed to plugin capability queries.
/// Uses a simple 1x1 convolution similar to fusilli's own integration tests.
inline flatbuffers::DetachedBuffer buildConvFpropGraph()
{
    using namespace hipdnn_frontend;
    using namespace hipdnn_frontend::graph;

    Graph graph;
    graph.set_name("Conv_Fprop_CapabilityTest")
         .set_compute_data_type(DataType::FLOAT)
         .set_io_data_type(DataType::FLOAT);

    // Dimensions matching fusilli test pattern
    const int64_t n = 16;   // batch
    const int64_t c = 128;  // in channels
    const int64_t h = 64;   // image height
    const int64_t w = 64;   // image width
    const int64_t k = 256;  // out channels
    const int64_t r = 1;    // filter height (1x1 conv)
    const int64_t s = 1;    // filter width

    // Input tensor: NCHW format with NCHW strides
    std::vector<int64_t> inputDims = {n, c, h, w};
    std::vector<int64_t> inputStrides = {c * h * w, h * w, w, 1};
    auto x = std::make_shared<TensorAttributes>(
        makeTensorAttributes("x", inputDims, inputStrides));
    x->set_uid(1);

    // Weight tensor: KCRS format
    std::vector<int64_t> weightDims = {k, c, r, s};
    std::vector<int64_t> weightStrides = {c * r * s, r * s, s, 1};
    auto wt = std::make_shared<TensorAttributes>(
        makeTensorAttributes("w", weightDims, weightStrides));
    wt->set_uid(2);

    // Build convolution operation (1x1 conv, no padding)
    ConvFpropAttributes convAttrs;
    convAttrs.set_padding({0, 0})
             .set_stride({1, 1})
             .set_dilation({1, 1});

    auto y = graph.conv_fprop(x, wt, convAttrs);

    // Output tensor: NKHW (same spatial dims for 1x1 stride-1 conv)
    std::vector<int64_t> outputDims = {n, k, h, w};
    std::vector<int64_t> outputStrides = {k * h * w, h * w, w, 1};
    y->set_dim(outputDims);
    y->set_stride(outputStrides);
    y->set_uid(3);
    y->set_output(true);

    // Validate and serialize
    auto result = graph.validate();
    if (result.code != ErrorCode::OK) {
        throw std::runtime_error("Failed to validate conv graph: " + result.get_message());
    }

    return graph.buildFlatbufferOperationGraph();
}

} // namespace hipdnn_integration_tests