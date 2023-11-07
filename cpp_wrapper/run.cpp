#include "triton.h"
// #include "cuTLASS.h"
// #include "cuBLAS.h"
#include <iostream>
#include <chrono>

int main() {
    const int M = 4; // Matrix dimensions (adjust as needed)
    const int N = 4;
    const int K = 4;

    // float A[M][K];
    // float B[K][N];
    // float C1[M][N];
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C1 = new float[M*N];
    // float* C2 = new float[M+N];
    // float* C3 = new float[M+N];

    for (int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            A[i*M+j] = i+j;
            std::cout<<A[i*M+j]<<" ";
        }
        std::cout<<"\n";
    }

    for (int i=0; i<K; i++){
        for(int j=0; j<N; j++){
            B[i*K+j] = i+j;
            std::cout<<B[i*K+j]<<" ";
        }
        std::cout<<"\n";
    }

    // Initialize matrices A and B (fill with data)

    Triton triton("matmul_tf32_64_32.ptx", "matmul_kernel_01234567891011", "leaky_relu");
    triton.forward(A, B, C1, M, N, K);

    for (int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            std::cout<<C1[i*M+j]<<" ";
        }
        std::cout<<"\n";
    }
    
    // auto start = std::chrono::high_resolution_clock::now();

    // // Perform the forward pass for each class and compare the results
    // for (int i = 0; i < 100; i++) {
    //     triton.forward(A, B, C1);
    //     cuTlass.forward(A, B, C2);
    //     cuBlas.forward(A, B, C3);
    // }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // // Compare C1, C2, and C3 and print the results
    // // ...

    // std::cout << "Forward pass time for each class: " << duration << " milliseconds" << std::endl;

    // // Clean up allocated memory (delete[] A, B, C1, C2, C3)

    // return 0;
}
