#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

const char* metal_src = R"(
#include <metal_stdlib>
using namespace metal;

kernel void softmax_kernel(
    device const float* input       [[buffer(0)]],
    device       float* output      [[buffer(1)]],
    uint2               tg_pos      [[threadgroup_position_in_grid]],
    uint2               t_pos       [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_max[1024];
    threadgroup float shared_sum[1024];

    // Один блок обробляє один рядок матриці
    uint idx = tg_pos.y * 1024 + t_pos.x;
    float val = input[idx];

    // Початкові значення
    shared_max[t_pos.x] = val;
    shared_sum[t_pos.x] = 1.0f;

    // Паралельна редукція (Дерево)
    for (uint s = 1024 / 2; s > 0; s >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t_pos.x < s) {
            float m1 = shared_max[t_pos.x];
            float m2 = shared_max[t_pos.x + s];
            float s1 = shared_sum[t_pos.x];
            float s2 = shared_sum[t_pos.x + s];

            float m_new = max(m1, m2);
            float s_new = s1 * exp(m1 - m_new) + s2 * exp(m2 - m_new);

            shared_max[t_pos.x] = m_new;
            shared_sum[t_pos.x] = s_new;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float final_max = shared_max[0];
    float final_sum = shared_sum[0];

    // Фінальний запис
    output[idx] = exp(val - final_max) / final_sum;
}
)";

at::Tensor softmax_metal_forward(at::Tensor input) {
    auto device = MTLCreateSystemDefaultDevice();
    auto commandQueue = [device newCommandQueue];
    
    // 1. Обов'язково перекидаємо на CPU, щоб Metal міг створити буфер, 
    // або працюємо з raw pointers. Найстабільніше для кастомних опів:
    auto input_cpu = input.to(at::kCPU).contiguous();
    auto output_cpu = at::empty_like(input_cpu);

    NSError *error = nil;
    NSString *source = [NSString stringWithUTF8String:metal_src];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    id<MTLFunction> function = [library newFunctionWithName:@"softmax_kernel"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    // 2. Створюємо Metal буфери напряму
    id<MTLBuffer> inBuffer = [device newBufferWithBytes:input_cpu.data_ptr<float>() 
                                                 length:input_cpu.nbytes() 
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuffer = [device newBufferWithBytes:output_cpu.data_ptr<float>() 
                                                  length:output_cpu.nbytes() 
                                                 options:MTLResourceStorageModeShared];

    auto commandBuffer = [commandQueue commandBuffer];
    auto encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inBuffer offset:0 atIndex:0];
    [encoder setBuffer:outBuffer offset:0 atIndex:1];
    
    uint32_t num_rows = input_cpu.numel() / 1024;
    MTLSize threadgroupsPerGrid = MTLSizeMake(1, num_rows, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(1024, 1, 1);
    
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // 3. Копіюємо результат назад
    memcpy(output_cpu.data_ptr<float>(), [outBuffer contents], output_cpu.nbytes());

    return output_cpu.to(input.device());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax_metal_forward, "Softmax forward (Metal)");
}