#pragma once
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

class Triton {
public:
    Triton(const std::string &module_file, 
    const std::string &kernel_name, 
    const std::string& activation);

    void forward(float const *ip,
                    float const *wt,
                    float *op,
                    int M,
                    int N,
                    int K);
private:
    CUfunction function_;
    std::string activation_;
    CUdevice dev_;
    CUcontext ctx_;
    CUmodule module_;
};
