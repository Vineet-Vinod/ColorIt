#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Stats {
  double median_ms;
  double mean_ms;
  double min_ms;
  double max_ms;
};

static Stats summarize(std::vector<double> samples_ms) {
  std::sort(samples_ms.begin(), samples_ms.end());
  const double median_ms = samples_ms[samples_ms.size() / 2];
  const double mean_ms =
      std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) / static_cast<double>(samples_ms.size());
  return Stats{
      median_ms,
      mean_ms,
      samples_ms.front(),
      samples_ms.back(),
  };
}

static NSString* shapeString(NSArray<NSNumber*>* shape) {
  return [NSString stringWithFormat:@"%@", shape];
}

static Stats benchLayout(id<MTLDevice> device, id<MTLCommandQueue> queue, bool nhwc, int warmup, int iterations) {
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];

    NSArray<NSNumber*>* inputShape = nhwc ? @[ @16, @336, @336, @259 ] : @[ @16, @259, @336, @336 ];
    NSArray<NSNumber*>* outputShape = inputShape;
    NSArray<NSNumber*>* weightShape = @[ @259, @259, @3, @3 ];
    NSArray<NSNumber*>* biasShape = @[ @259 ];
    NSArray<NSNumber*>* biasBroadcastShape = nhwc ? @[ @1, @1, @1, @259 ] : @[ @1, @259, @1, @1 ];

    MPSGraphTensor* inputTensor = [graph placeholderWithShape:inputShape dataType:MPSDataTypeFloat32 name:@"input"];
    MPSGraphTensor* weightTensor = [graph placeholderWithShape:weightShape dataType:MPSDataTypeFloat32 name:@"weight"];
    MPSGraphTensor* biasTensor = [graph placeholderWithShape:biasShape dataType:MPSDataTypeFloat32 name:@"bias"];

    MPSGraphConvolution2DOpDescriptor* desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1
                                                         strideInY:1
                                                   dilationRateInX:1
                                                   dilationRateInY:1
                                                            groups:1
                                                       paddingLeft:1
                                                      paddingRight:1
                                                        paddingTop:1
                                                     paddingBottom:1
                                                      paddingStyle:MPSGraphPaddingStyleExplicit
                                                        dataLayout:(nhwc ? MPSGraphTensorNamedDataLayoutNHWC
                                                                         : MPSGraphTensorNamedDataLayoutNCHW)
                                                     weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    MPSGraphTensor* convTensor =
        [graph convolution2DWithSourceTensor:inputTensor weightsTensor:weightTensor descriptor:desc name:@"conv"];
    MPSGraphTensor* reshapedBias =
        [graph reshapeTensor:biasTensor withShape:biasBroadcastShape name:@"bias_reshape"];
    MPSGraphTensor* outputTensor =
        [graph additionWithPrimaryTensor:convTensor secondaryTensor:reshapedBias name:@"bias_add"];

    MPSGraphDevice* graphDevice = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraphTensorShapedTypeDictionary* compileFeeds = @{
      inputTensor : [[MPSGraphShapedType alloc] initWithShape:inputShape dataType:MPSDataTypeFloat32],
      weightTensor : [[MPSGraphShapedType alloc] initWithShape:weightShape dataType:MPSDataTypeFloat32],
      biasTensor : [[MPSGraphShapedType alloc] initWithShape:biasShape dataType:MPSDataTypeFloat32],
    };

    MPSGraphExecutable* executable =
        [graph compileWithDevice:graphDevice
                           feeds:compileFeeds
                   targetTensors:@[ outputTensor ]
                targetOperations:nil
           compilationDescriptor:nil];

    const uint64_t inputElements = 16ull * 259ull * 336ull * 336ull;
    const uint64_t outputElements = inputElements;
    const uint64_t weightElements = 259ull * 259ull * 3ull * 3ull;
    const uint64_t biasElements = 259ull;
    const NSUInteger inputBytes = static_cast<NSUInteger>(inputElements * sizeof(float));
    const NSUInteger outputBytes = static_cast<NSUInteger>(outputElements * sizeof(float));
    const NSUInteger weightBytes = static_cast<NSUInteger>(weightElements * sizeof(float));
    const NSUInteger biasBytes = static_cast<NSUInteger>(biasElements * sizeof(float));

    id<MTLBuffer> inputBuffer = [device newBufferWithLength:inputBytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> weightBuffer = [device newBufferWithLength:weightBytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> biasBuffer = [device newBufferWithLength:biasBytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:outputBytes options:MTLResourceStorageModePrivate];

    MPSGraphTensorData* inputData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer shape:inputShape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* weightData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:weightBuffer shape:weightShape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* biasData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:biasBuffer shape:biasShape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* outputData =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer shape:outputShape dataType:MPSDataTypeFloat32];

    NSArray<MPSGraphTensorData*>* inputs = @[ inputData, weightData, biasData ];
    NSArray<MPSGraphTensorData*>* results = @[ outputData ];

    MPSGraphExecutableExecutionDescriptor* execDesc = [[MPSGraphExecutableExecutionDescriptor alloc] init];
    execDesc.waitUntilCompleted = YES;

    for (int i = 0; i < warmup; ++i) {
      [executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:results executionDescriptor:execDesc];
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      const auto started = std::chrono::steady_clock::now();
      [executable runWithMTLCommandQueue:queue inputsArray:inputs resultsArray:results executionDescriptor:execDesc];
      const auto finished = std::chrono::steady_clock::now();
      const double elapsed_ms =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(finished - started).count();
      samples_ms.push_back(elapsed_ms);
    }

    Stats stats = summarize(samples_ms);
    printf("layout=%s input_shape=%s median_ms=%.4f mean_ms=%.4f min_ms=%.4f max_ms=%.4f\n",
           nhwc ? "NHWC" : "NCHW",
           shapeString(inputShape).UTF8String,
           stats.median_ms,
           stats.mean_ms,
           stats.min_ms,
           stats.max_ms);
    return stats;
  }
}

} // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      fprintf(stderr, "No Metal device available\n");
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      fprintf(stderr, "Failed to create command queue\n");
      return 1;
    }

    int warmup = 5;
    int iterations = 10;
    if (argc >= 2) {
      warmup = std::max(1, atoi(argv[1]));
    }
    if (argc >= 3) {
      iterations = std::max(1, atoi(argv[2]));
    }

    printf("device=%s warmup=%d iterations=%d\n", device.name.UTF8String, warmup, iterations);
    benchLayout(device, queue, false, warmup, iterations);
    benchLayout(device, queue, true, warmup, iterations);
  }
  return 0;
}
