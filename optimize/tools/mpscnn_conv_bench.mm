#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

static std::vector<float> makeWeights();
static std::vector<float> makeBiases();
static MPSCNNConvolutionDescriptor* makeConvDescriptor();

@interface BaseConvolutionDataSource : NSObject <MPSCNNConvolutionDataSource> {
 @protected
  MPSCNNConvolutionDescriptor* _descriptor;
  std::vector<float> _weights;
  std::vector<float> _biases;
  NSString* _label;
}

- (instancetype)initWithLabel:(NSString*)label;

@end

@implementation BaseConvolutionDataSource

- (instancetype)initWithLabel:(NSString*)label {
  self = [super init];
  if (self == nil) {
    return nil;
  }
  _descriptor = makeConvDescriptor();
  _weights = makeWeights();
  _biases = makeBiases();
  _label = [label copy];
  return self;
}

- (id)copyWithZone:(NSZone*)zone {
  return [[[self class] allocWithZone:zone] initWithLabel:_label];
}

- (MPSDataType)dataType {
  return MPSDataTypeFloat32;
}

- (MPSCNNConvolutionDescriptor*)descriptor {
  return _descriptor;
}

- (void*)weights {
  return _weights.data();
}

- (float*)biasTerms {
  return _biases.data();
}

- (BOOL)load {
  return YES;
}

- (void)purge {
}

- (NSString*)label {
  return _label;
}

- (MPSCNNConvolutionWeightsLayout)weightsLayout {
  return MPSCNNConvolutionWeightsLayoutOHWI;
}

@end

@interface Float32KernelDataSource : BaseConvolutionDataSource
@end

@implementation Float32KernelDataSource

- (MPSDataType)kernelWeightsDataType {
  return MPSDataTypeFloat32;
}

@end

constexpr NSUInteger kBatch = 16;
constexpr NSUInteger kWidth = 336;
constexpr NSUInteger kHeight = 336;
constexpr NSUInteger kChannels = 259;
constexpr NSUInteger kKernel = 3;

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

static std::vector<float> makeWeights() {
  std::vector<float> weights(kChannels * kChannels * kKernel * kKernel);
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = std::sin(static_cast<float>(i) * 0.001f);
  }
  return weights;
}

static std::vector<float> makeBiases() {
  std::vector<float> bias(kChannels);
  for (size_t i = 0; i < bias.size(); ++i) {
    bias[i] = std::cos(static_cast<float>(i) * 0.01f);
  }
  return bias;
}

static MPSCNNConvolutionDescriptor* makeConvDescriptor() {
  MPSCNNConvolutionDescriptor* desc =
      [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kKernel
                                                              kernelHeight:kKernel
                                                      inputFeatureChannels:kChannels
                                                     outputFeatureChannels:kChannels];
  desc.strideInPixelsX = 1;
  desc.strideInPixelsY = 1;
  desc.groups = 1;
  desc.dilationRateX = 1;
  desc.dilationRateY = 1;
  return desc;
}

static MPSImageDescriptor* makeImageDescriptor() {
  MPSImageDescriptor* desc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                                            width:kWidth
                                                                           height:kHeight
                                                                  featureChannels:kChannels
                                                                   numberOfImages:kBatch
                                                                            usage:MTLTextureUsageShaderRead |
                                                                                  MTLTextureUsageShaderWrite];
  desc.storageMode = MTLStorageModePrivate;
  return desc;
}

struct VariantResult {
  std::string name;
  Stats stats;
};

static MPSImageBatch* makeSingleImageBatch(id<MTLDevice> device, MPSImage** parentImage) {
  MPSImage* parent = [[MPSImage alloc] initWithDevice:device imageDescriptor:makeImageDescriptor()];
  (void)parent.texture;
  *parentImage = parent;
  return [parent batchRepresentation];
}

static MPSImage* makeParentBatchImage(id<MTLDevice> device) {
  MPSImage* image = [[MPSImage alloc] initWithDevice:device imageDescriptor:makeImageDescriptor()];
  (void)image.texture;
  return image;
}

static Stats runConvolution(id<MTLDevice> device,
                            id<MTLCommandQueue> queue,
                            MPSCNNConvolution* convolution,
                            int warmup,
                            int iterations) {
  @autoreleasepool {
    MPSImage* sourceParent = nil;
    MPSImage* destinationParent = nil;
    MPSImageBatch* sourceBatch = makeSingleImageBatch(device, &sourceParent);
    MPSImageBatch* destinationBatch = makeSingleImageBatch(device, &destinationParent);

    for (int i = 0; i < warmup; ++i) {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        [convolution encodeBatchToCommandBuffer:commandBuffer
                                   sourceImages:sourceBatch
                              destinationImages:destinationBatch];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
      }
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        const auto started = std::chrono::steady_clock::now();
        [convolution encodeBatchToCommandBuffer:commandBuffer
                                   sourceImages:sourceBatch
                              destinationImages:destinationBatch];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto finished = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(finished - started).count();
        samples_ms.push_back(elapsed_ms);
      }
    }

    return summarize(samples_ms);
  }
}

static Stats runConvolutionParentImages(id<MTLDevice> device,
                                        id<MTLCommandQueue> queue,
                                        MPSCNNConvolution* convolution,
                                        int warmup,
                                        int iterations) {
  @autoreleasepool {
    MPSImage* sourceImage = makeParentBatchImage(device);
    MPSImage* destinationImage = makeParentBatchImage(device);

    convolution.clipRect = MTLRegionMake3D(0, 0, 0, kWidth, kHeight, kBatch);

    for (int i = 0; i < warmup; ++i) {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        [convolution encodeToCommandBuffer:commandBuffer
                               sourceImage:sourceImage
                          destinationImage:destinationImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
      }
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        const auto started = std::chrono::steady_clock::now();
        [convolution encodeToCommandBuffer:commandBuffer
                               sourceImage:sourceImage
                          destinationImage:destinationImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto finished = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(finished - started).count();
        samples_ms.push_back(elapsed_ms);
      }
    }

    return summarize(samples_ms);
  }
}

static VariantResult benchDataSourceDefault(id<MTLDevice> device, id<MTLCommandQueue> queue, int warmup, int iterations) {
  BaseConvolutionDataSource* dataSource = [[BaseConvolutionDataSource alloc] initWithLabel:@"data_source_default"];
  MPSCNNConvolution* convolution = [[MPSCNNConvolution alloc] initWithDevice:device weights:dataSource];
  if (convolution == nil) {
    fprintf(stderr, "Failed to create MPSCNNConvolution for data_source_default\n");
    std::exit(1);
  }
  convolution.accumulatorPrecisionOption = MPSNNConvolutionAccumulatorPrecisionOptionFloat;
  convolution.edgeMode = MPSImageEdgeModeZero;
  convolution.offset = MPSOffset{ static_cast<NSInteger>(kKernel / 2), static_cast<NSInteger>(kKernel / 2), 0 };
  return VariantResult{ "data_source_default", runConvolution(device, queue, convolution, warmup, iterations) };
}

static VariantResult benchDataSourceFloat32Kernel(id<MTLDevice> device,
                                                  id<MTLCommandQueue> queue,
                                                  int warmup,
                                                  int iterations) {
  Float32KernelDataSource* dataSource =
      [[Float32KernelDataSource alloc] initWithLabel:@"data_source_float32_kernel"];
  MPSCNNConvolution* convolution = [[MPSCNNConvolution alloc] initWithDevice:device weights:dataSource];
  if (convolution == nil) {
    fprintf(stderr, "Failed to create MPSCNNConvolution for data_source_float32_kernel\n");
    std::exit(1);
  }
  convolution.accumulatorPrecisionOption = MPSNNConvolutionAccumulatorPrecisionOptionFloat;
  convolution.edgeMode = MPSImageEdgeModeZero;
  convolution.offset = MPSOffset{ static_cast<NSInteger>(kKernel / 2), static_cast<NSInteger>(kKernel / 2), 0 };
  return VariantResult{ "data_source_float32_kernel", runConvolution(device, queue, convolution, warmup, iterations) };
}

static VariantResult benchRawPointerInit(id<MTLDevice> device, id<MTLCommandQueue> queue, int warmup, int iterations) {
  const std::vector<float> weights = makeWeights();
  const std::vector<float> biases = makeBiases();
  MPSCNNConvolutionDescriptor* descriptor = makeConvDescriptor();
  MPSCNNConvolution* convolution = [[MPSCNNConvolution alloc] initWithDevice:device
                                                       convolutionDescriptor:descriptor
                                                               kernelWeights:weights.data()
                                                                   biasTerms:biases.data()
                                                                       flags:MPSCNNConvolutionFlagsNone];
  if (convolution == nil) {
    fprintf(stderr, "Failed to create MPSCNNConvolution for raw_pointer_init\n");
    std::exit(1);
  }
  convolution.accumulatorPrecisionOption = MPSNNConvolutionAccumulatorPrecisionOptionFloat;
  convolution.edgeMode = MPSImageEdgeModeZero;
  convolution.offset = MPSOffset{ static_cast<NSInteger>(kKernel / 2), static_cast<NSInteger>(kKernel / 2), 0 };
  return VariantResult{ "raw_pointer_init", runConvolution(device, queue, convolution, warmup, iterations) };
}

static VariantResult benchRawPointerParentImage(id<MTLDevice> device,
                                                id<MTLCommandQueue> queue,
                                                int warmup,
                                                int iterations) {
  const std::vector<float> weights = makeWeights();
  const std::vector<float> biases = makeBiases();
  MPSCNNConvolutionDescriptor* descriptor = makeConvDescriptor();
  MPSCNNConvolution* convolution = [[MPSCNNConvolution alloc] initWithDevice:device
                                                       convolutionDescriptor:descriptor
                                                               kernelWeights:weights.data()
                                                                   biasTerms:biases.data()
                                                                       flags:MPSCNNConvolutionFlagsNone];
  if (convolution == nil) {
    fprintf(stderr, "Failed to create MPSCNNConvolution for raw_pointer_parent_image\n");
    std::exit(1);
  }
  convolution.accumulatorPrecisionOption = MPSNNConvolutionAccumulatorPrecisionOptionFloat;
  convolution.edgeMode = MPSImageEdgeModeZero;
  convolution.offset = MPSOffset{ static_cast<NSInteger>(kKernel / 2), static_cast<NSInteger>(kKernel / 2), 0 };
  return VariantResult{ "raw_pointer_parent_image",
                        runConvolutionParentImages(device, queue, convolution, warmup, iterations) };
}

static void printResult(const VariantResult& result) {
  printf("variant=%s median_ms=%.4f mean_ms=%.4f min_ms=%.4f max_ms=%.4f\n",
         result.name.c_str(),
         result.stats.median_ms,
         result.stats.mean_ms,
         result.stats.min_ms,
         result.stats.max_ms);
}

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

    int warmup = 3;
    int iterations = 5;
    if (argc >= 2) {
      warmup = std::max(1, atoi(argv[1]));
    }
    if (argc >= 3) {
      iterations = std::max(1, atoi(argv[2]));
    }

    printf("device=%s warmup=%d iterations=%d batch=%lu shape=[%lu,%lu,%lu,%lu] kernel=%lux%lu\n",
           device.name.UTF8String,
           warmup,
           iterations,
           static_cast<unsigned long>(kBatch),
           static_cast<unsigned long>(kBatch),
           static_cast<unsigned long>(kChannels),
           static_cast<unsigned long>(kHeight),
           static_cast<unsigned long>(kWidth),
           static_cast<unsigned long>(kKernel),
           static_cast<unsigned long>(kKernel));

    printResult(benchDataSourceDefault(device, queue, warmup, iterations));
    printResult(benchDataSourceFloat32Kernel(device, queue, warmup, iterations));
    printResult(benchRawPointerInit(device, queue, warmup, iterations));
    printResult(benchRawPointerParentImage(device, queue, warmup, iterations));
  }
  return 0;
}
