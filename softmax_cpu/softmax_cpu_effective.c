#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

void softmax_cpu(const float *restrict input, float *restrict output, int N) {
    float max_x = input[0];
    for (int i = 1; i < N; i++) {
        if (input[i] > max_x) {
            max_x = input[i];
        }
    }

    float denominator_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = expf(input[i] - max_x);
        output[i] = e;
        denominator_sum += e;
    }

    float inverse_denominator_sum = 1.0f / denominator_sum;
    for (int i = 0; i < N; i++) {
        output[i] *= inverse_denominator_sum;
    }
}

int main() {
    int N = 10000000;
    float *input = (float*)malloc(N * sizeof(float));
    float *output = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = (float)(i % 100) / 10.0f;
    }

    softmax_cpu(input, output, N);


    struct timespec ts1, tw1; // both C11 and POSIX
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts1); // POSIX
    clock_gettime(CLOCK_MONOTONIC, &tw1); 
    double start_time = clock();
    
    softmax_cpu(input, output, N);
    
    struct timespec ts2, tw2;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts2);
    clock_gettime(CLOCK_MONOTONIC, &tw2);
    clock_t end_time = clock();
    
    double dur = 1000.0 * (end_time - start_time) / CLOCKS_PER_SEC;
    double posix_dur = 1000.0 * ts2.tv_sec + 1e-6 * ts2.tv_nsec
                           - (1000.0 * ts1.tv_sec + 1e-6 * ts1.tv_nsec);
    double posix_wall = 1000.0 * tw2.tv_sec + 1e-6 * tw2.tv_nsec
                            - (1000.0 * tw1.tv_sec + 1e-6 * tw1.tv_nsec);
    
    printf("CPU time used (per clock()): %.2f ms\n", dur);
    printf("CPU time used (per clock_gettime()): %.2f ms\n", posix_dur);
    printf("Wall time passed: %.2f ms\n", posix_wall);
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%e ", output[i]);
    }
    printf("\n");

    free(input);
    free(output);
    return 0;
}