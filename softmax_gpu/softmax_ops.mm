#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

at::Tensor softmax_metal_forward(at::Tensor input) {
    auto device = MTLCreateSystemDefaultDevice();
    auto commandQueue = [device newCommandQueue];
    
    auto output = at::empty_like(input);

    NSError *error = nil;
    NSString *path = [[NSBundle mainBundle] pathForResource:@"softmax_kernel" ofType:@"metal"];
    NSString *source = [NSString stringWithContentsOfFile:@"softmax_kernel.metal" encoding:NSUTF8StringEncoding error:&error];
    
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"softmax_kernel"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    auto commandBuffer = [commandQueue commandBuffer];
    auto encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    [encoder setBytes:input.data_ptr<float>() length:input.nbytes() atIndex:0];
    [encoder setBytes:output.data_ptr<float>() length:output.nbytes() atIndex:1];

    uint32_t num_rows = input.numel() / 1024;
    
    MTLSize threadgroupsPerGrid = MTLSizeMake(1, num_rows, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(1024, 1, 1);
    
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_metal_forward, "Softmax forward (Metal)");
}