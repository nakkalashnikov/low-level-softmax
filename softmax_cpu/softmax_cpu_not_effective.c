#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

void softmax_cpu(const float *restrict input, float *restrict output, int N) {
    float max_x = input[0];
    float denominator_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float input_i = input[i];
        if (input_i > max_x) {
            denominator_sum = (denominator_sum * expf(max_x - input_i)) + 1.0f;
            max_x = input_i;
        } else {
            denominator_sum += expf(input_i - max_x);
        }
    }

    float inverse_denominator_sum = 1.0f / denominator_sum;
    for (int i = 0; i < N; i++) {
        float input_i = input[i];
        float e = expf(input_i - max_x);
        output[i] = e * inverse_denominator_sum;
    }
}

int main() {
    int N = 10000000;
    float *input = (float*)malloc(N * sizeof(float));
    float *output = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = (float)(i % 100) / 10.f;
    }

    softmax_cpu(input, output, N);

    double start_time = get_time_ms();
    
    softmax_cpu(input, output, N);
    
    double end_time = get_time_ms();
    double elapsed = end_time - start_time;

    printf("Time for pure C: %.4f ms\n", elapsed);
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%e ", output[i]);
    }
    printf("\n");

    free(input);
    free(output);
    return 0;
}