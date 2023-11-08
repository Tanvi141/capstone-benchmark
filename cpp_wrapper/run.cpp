#include "triton.h"
// #include "cuTLASS.h"
#include "cublas.h"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    const int M = 2; // Matrix dimensions (adjust as needed)
    const int N = 3;
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
            A[i*M+j] = i+j*2;
            A[0] = 5;
            // std::cout<<A[i*M+j]<<" ";
        }
    }

    for (int i=0; i<K; i++){
        for(int j=0; j<N; j++){
            B[i*K+j] = i*2+j;
            B[1] = -1;
            // std::cout<<B[i*K+j]<<" ";
        }
    }

    std::cout<<"A\n";
    for (int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            std::cout<<A[i+j*K]<<" ";
        }
        std::cout<<"\n";
    }

    std::cout<<"B\n";
    for (int i=0; i<K; i++){
        for(int j=0; j<N; j++){
            std::cout<<B[i+j*N]<<" ";
        }
        std::cout<<"\n";
    }

    for (int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            C1[i*M+j] = 0;
            // std::cout<<C1[i*M+j]<<" ";
        }
        // std::cout<<"\n";
    }

    // Initialize matrices A and B (fill with data)

    // Triton triton("matmul_tf32_64_32.ptx", "matmul_kernel_01234567891011", "leaky_relu");
    // Triton triton("matmul_kernel.ptx", "matmul_kernel_0d1d2d3d4d5c6c7c8d9c10d11c", "leaky_relu");
    // triton.forward(A, B, C1, M, N, K);
    
    int poss[6] = {
        std::stoi(argv[1]), 
        std::stoi(argv[2]), 
        std::stoi(argv[3]), 
        std::stoi(argv[4]), 
        std::stoi(argv[5]), 
        std::stoi(argv[6])};

    Cublas cublas("leaky_relu");
    cublas.forward(A, B, C1, M, N, K, poss);
    std::cout<<"C\n";
    for (int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            std::cout<<C1[i+j*N]<<" ";
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
