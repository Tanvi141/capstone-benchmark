#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cublas_v2.h>

#include "cublas.h"

Cublas::Cublas(const std::string& activation) {
    activation_ = activation;
    cublasCreate(&handle_);
    cudaStreamCreate(&stream_);
    cublasSetStream(handle_, stream_);
}

void Cublas::forward(float const *ip, //A
                    float const *wt, // B
                    float *op, // C
                    int M,
                    int N,
                    int K){
    
    float* A;
    float* B;
    float* C;
    cudaMalloc((void**)&A, M * K * sizeof(float));
    cudaMalloc((void**)&B, K * N * sizeof(float));
    cudaMalloc((void**)&C, M * N * sizeof(float));

    cudaMemcpy(A, ip, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, wt, K * N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    // cublasStatus_t status = cublasGemmEx(
    //     handle_,
    //     CUBLAS_OP_N, CUBLAS_OP_N, // No transpose for A and B
    //     N, M, K, // Matrix dimensions
    //     &alpha,
    //     B, CUDA_R_32F, N, // A matrix
    //     A, CUDA_R_32F, K, // B matrix
    //     &beta,
    //     C, CUDA_R_32F, N, // Output C matrix
    //     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP // Data types and operation
    // );

    cublasStatus_t status = cublasGemmEx(
        handle_,
        CUBLAS_OP_N, CUBLAS_OP_N, // No transpose for A and B
        M, N, K, // Matrix dimensions
        &alpha,
        B, CUDA_R_32F, N, // A matrix
        A, CUDA_R_32F, K, // B matrix
        &beta,
        C, CUDA_R_32F, N, // Output C matrix
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP // Data types and operation
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmEx failed with error code: " << status << std::endl;
        // Handle the error or print more details if needed.
    }

    cudaMemcpy(op, C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream_);

}


// cuModuleUnload(module);

