#pragma once
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <iostream>

class Cublas {
public:
    Cublas(const std::string& activation);

    void forward(float const *ip,
                    float const *wt,
                    float *op,
                    int M,
                    int N,
                    int K);
private:
    std::string activation_;
    cublasHandle_t handle_;
    cudaStream_t stream_;
};
