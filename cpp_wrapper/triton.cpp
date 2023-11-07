#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

using namespace std;

int main(){
// Define the kernel launch parameters (you may need to adjust these)
int M = 1024;  // Replace with your desired values
int N = 1024;
int K = 64;

// Load the PTX module
//char module_file[] = "../.triton/cache/6aeab1ae2060c2a7404194cc1be14bdf/matmul_kernel.ptx";
char module_file[] = "matmul_tf32_64_32.ptx";
//char kernel_name[] = "matmul_kernel_0d1d2d3d4d5c6c7c8d9c10d11c";
char kernel_name[] = "matmul_kernel_01234567891011";
	
CUresult err1 = cuInit(0);  // Initialize the CUDA driver API

 CUdevice dev;
  CUcontext ctx;
  //CUstream stream;
  ////CUdeviceptr A, B, C;
  //CUresult err = 0;
  //cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);

CUmodule module;
CUfunction function;


if (err1 != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to initialize the CUDA driver API\n");
    exit(1);
}
fprintf(stdout, "Init success\n");

CUresult err2 = cuModuleLoad(&module, module_file);

if (err2 != CUDA_SUCCESS) {
    cout<<err2<<"\n";
    fprintf(stderr, "Failed to load the CUDA module\n");
    exit(1);
}
fprintf(stdout, "Module load success\n");

CUresult err3 = cuModuleGetFunction(&function, module, kernel_name);

if (err3 != CUDA_SUCCESS) {
    fprintf(stderr, "Failed to get the CUDA function\n");
    exit(1);
}
fprintf(stdout, "Function load success\n");

// Calculate the grid size
int bm = 32;
int bn = 64;
int bk = 32;
int gm = 8;
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
cuMemAlloc(&A, M * K * 2);
cuMemAlloc(&B, K * N * 2);
cuMemAlloc(&C, M * N * 4);
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
for (int i = 0; i < 1; i++) {
    err = cuLaunchKernel(function, grid, 1, 1, 128, 1, 1, 24576, hStream, args, NULL);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to launch the kernel\n");
        exit(1);
    }
    fprintf(stdout, "Launch success\n");
/*
    for (int i = 0; i < M; i++) {
	    for (int j = 0; j < K; j++) {
		    std::cout << A[i][j] << " ";
	    }
	    std::cout << "\n";
    }

    for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                    std::cout << B[i][j] << " ";
            }
            std::cout << "\n";
    }

    for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                    std::cout << C[i][j] << " ";
            }
            std::cout << "\n";
    }
    */
}

// Free allocated memory
//cuMemFree(args);

// Unload the module (not shown in the original code)
cuModuleUnload(module);
}
