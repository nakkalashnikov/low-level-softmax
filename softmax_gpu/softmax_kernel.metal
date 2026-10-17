#include <metal_stdlib>
using namespace metal;

kernel void softmax_kernel(
    device const float* input  [[buffer(0)]],
    device const float* output [[buffer(1)]],
    uint2               tg_pos [[threadgroup_position_in_grid]]
    uint2               t_pos  [[thread_position_in_threadgroup]]
    uint2               t_grid [[thread_position_in_grid]]
) {
    threadgroup float shared_max[1024];
    threadgroup float shared_sum[1024];

    int idx = tg_pos.y * 1024 + t_pos.x;
    float max_val = input[idx];
    float denominator_sum = 1.0f;

    shared_max[t_pos.x] = max_val;
    shared_sum[t_pos.x] = denominator_sum;

    for (uint s = 1024 / 2; s > 0; s >>= 1 ) {
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

    output[idx] = (exp(input[idx] - final_max)) / final_sum;
}