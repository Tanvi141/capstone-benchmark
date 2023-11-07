#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#include "triton.h"

Triton::Triton(const std::string &module_file, 
                const std::string &kernel_name, 
                const std::string& activation) {
    activation_ = activation;

    CUresult err1 = cuInit(0);  // Initialize the CUDA driver API
    if (err1 != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to initialize the CUDA driver API\n");
        exit(1);
    }
    fprintf(stdout, "Init success\n");
    cuDeviceGet(&dev_, 0);
    cuCtxCreate(&ctx_, 0, dev_);

    CUresult err2 = cuModuleLoad(&module_, module_file.c_str());

    if (err2 != CUDA_SUCCESS) {
        std::cout<<err2<<"\n";
        fprintf(stderr, "Failed to load the CUDA module\n");
        exit(1);
    }
    fprintf(stdout, "Module load success\n");

    CUresult err3 = cuModuleGetFunction(&function_, module_, kernel_name.c_str());

    if (err3 != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get the CUDA function\n");
        exit(1);
    }
    fprintf(stdout, "Function load success\n");

}

void Triton::forward(float const *ip, //A
                    float const *wt, // B
                    float *op, // C
                    int M,
                    int N,
                    int K){
    // Calculate the grid size
    int bm = 1;
    int bn = 1;
    int bk = 1;
    int gm = 1;
    int one = 1;
    int grid = (int)(M / bm) * (int)(N / bn);

    // Allocate memory for args (assuming args is a CUdeviceptr*)
    void* args[16];

    // Set the arguments in args (assuming args is an array of CUdeviceptr)
    // Replace these with your actual data pointers
    //CUdeviceptr arg1 = ...;  // Argument 1
    //CUdeviceptr arg2 = ...;  // Argument 2
    //CUdeviceptr arg3 = ...;  // Argument 3
    CUdeviceptr A, B, C;
    //  = new CUdeviceptr();
    // CUdeviceptr* B = new CUdeviceptr();
    // CUdeviceptr* C = new CUdeviceptr();
    cuMemAlloc(&A, M * K * sizeof(float));
    cuMemAlloc(&B, K * N * sizeof(float));
    cuMemAlloc(&C, M * N * sizeof(float));
    cuMemcpyHtoD(A, ip, M * K * sizeof(float));
    cuMemcpyHtoD(B, wt, K * N * sizeof(float));


    args[0] = &A;
    args[1] = &B;
    args[2] = &C;
    args[3] = &M;
    args[4] = &N;
    args[5] = &K;
    args[6] = &K;
    args[7] = &one;
    args[8] = &N;
    args[9] = &one;
    args[10] = &N;
    args[11] = &one;
    args[12] = &bm;
    args[13] = &bn;
    args[14] = &bk;
    args[15] = &gm;


    // Launch the kernel 10 times
    CUstream hStream;  // You may need to create a CUDA stream
    cuStreamCreate(&hStream, 0);
    cuStreamSynchronize(hStream);
    CUresult err;

    err = cuLaunchKernel(function_, grid, 1, 1, 128, 1, 1, 24576, hStream, args, NULL);

    if (err != CUDA_SUCCESS) {
        std::cout<<err<<"\n";
        fprintf(stderr, "Failed to launch the kernel\n");
        exit(1);
    }
    fprintf(stdout, "Launch success\n");

    cuMemcpyDtoH(op, C, M * N * sizeof(float));

    cuMemFree(A);
    cuMemFree(B);
    cuMemFree(C);
}


// cuModuleUnload(module);

