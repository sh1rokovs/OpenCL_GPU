#include <CL/cl.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include "../../../init.h"

void SequentialMatrixMultiplication(const float* a, const float* b, float* c, const size_t m, const size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
        {
            float res = 0.0;
            for (size_t k = 0; k < m; ++k)
                res += a[i * m + k] * b[k * n + j];
            c[i * n + j] = res;
        }
}

void ParallelMatrixMultiplication(const float* a, const float* b, float* c, const size_t m, const size_t n) {
    int num_tr = omp_get_max_threads();
    num_tr = n < num_tr ? n : num_tr;
#pragma omp parallel num_threads(num_tr)
    {
#pragma omp for schedule(static, n / num_tr)
        for (long long i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                float res = 0.0;
                for (size_t k = 0; k < m; ++k)
                    res += a[i * m + k] * b[k * n + j];
                c[i * n + j] = res;
            }
    }
}

void ParallelOpenCLMatrixMultiplication(const char* path, const size_t group, float* a, 
                                        float* b, float* c, const unsigned int m, const unsigned int n) {
    size_t group_size[] = { group, group };
    size_t size_n[] = { n, n };
    OCLInitialization pr(path, 2, group_size, size_n);
    pr.AddKernel("MatrixMultiplication");
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, n * m);
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, m * n);
    pr.AddBuffer<float>(CL_MEM_WRITE_ONLY, n * n);
    pr.WriteElementsToBuffer(0, n * m, a);
    pr.WriteElementsToBuffer(1, m * n, b);
    pr.SetKernelArg(pr.GetKernel(0), 0, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 1, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(2));
    pr.SetKernelArg(pr.GetKernel(0), 3, &m);
    pr.SetKernelArg(pr.GetKernel(0), 4, &n);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU float time - " << end - start << std::endl;
    pr.ReadElementsFromBuffer(2, n * n, c);
}

void ParallelOpenCLMatrixMultiplicationOptimized(const char* path, const size_t group, float* a,
                                                 float* b, float* c, const unsigned int m, const unsigned int n) {
    size_t group_size[] = { group, group };
    size_t size_n[] = { n, n };
    OCLInitialization pr(path, 2, group_size, size_n);
    pr.AddKernel("MatrixMultiplicationBlock");
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, n * m);
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, m * n);
    pr.AddBuffer<float>(CL_MEM_WRITE_ONLY, n * n);
    pr.WriteElementsToBuffer(0, n * m, a);
    pr.WriteElementsToBuffer(1, m * n, b);
    pr.SetKernelArg(pr.GetKernel(0), 0, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 1, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(2));
    pr.SetKernelArg(pr.GetKernel(0), 3, &m);
    pr.SetKernelArg(pr.GetKernel(0), 4, &n);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU optimized float time - " << end - start << std::endl;
    pr.ReadElementsFromBuffer(2, n * n, c);
}

void ParallelOpenCLMatrixMultiplicationImage(const char* path, const size_t group, float* a, 
                                             float* b, float* c, const unsigned int m, const unsigned int n) {
    size_t group_size[] = { group, group, 1 };
    size_t size_n[] = { n, n, 1 };
    OCLInitialization pr(path, 3, group_size, size_n);
    pr.AddKernel("MatrixMultiplicationImage");
    pr.AddImage<float>(CL_MEM_READ_ONLY, m, n);
    pr.AddImage<float>(CL_MEM_READ_ONLY, n, m);
    pr.AddImage<float>(CL_MEM_WRITE_ONLY, n, n);
    pr.WriteElementsToImage(0, m, n, a);
    pr.WriteElementsToImage(1, n, m, b);
    pr.SetKernelArg(pr.GetKernel(0), 0, &pr.GetImage(0));
    pr.SetKernelArg(pr.GetKernel(0), 1, &pr.GetImage(1));
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetImage(2));
    pr.SetKernelArg(pr.GetKernel(0), 3, &m);
    pr.SetKernelArg(pr.GetKernel(0), 4, &n);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU optimized by image float time - " << end - start << std::endl;
    pr.ReadElementsFromImage(2, n, n, c);
}

bool IsEqual(const float* a, const float* b, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > 0.01) {
            //std::cout << a[i] << "   " << b[i] << std::endl;
            printf("%d - %lf   %lf\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    //const unsigned int size = 1024;
    const unsigned int m = 2048;
    const unsigned int n = 1024;

    // init float
    float* a_seq = new float[n * m];
    float* b_seq = new float[n * m];
    float* c_seq = new float[n * n];

    float* a_par_cpu = new float[n * m];
    float* b_par_cpu = new float[n * m];
    float* c_par_cpu = new float[n * n];

    float* a_par_gpu = new float[n * m];
    float* b_par_gpu = new float[n * m];
    float* c_par_gpu = new float[n * n];

    float* a_par_gpu_opt = new float[n * m];
    float* b_par_gpu_opt = new float[n * m];
    float* c_par_gpu_opt = new float[n * n];

    float* a_par_gpu_img = new float[n * m];
    float* b_par_gpu_img = new float[n * m];
    float* c_par_gpu_img = new float[n * n];

    float lower_bound = -10.0;
    float upper_bound = 10.0;
    std::cout << "GPU: NVIDIA Geforce 1050 TI" << std::endl;
    std::cout << "CPU: Intel Core I5" << std::endl;
    std::cout << "Generation start" << std::endl;
    std::uniform_real_distribution<float> unif_f(lower_bound, upper_bound);
    std::default_random_engine re;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float num_f = unif_f(re);

            a_seq[i * n + j] = a_par_cpu[i * n + j] = a_par_gpu[i * n + j] = a_par_gpu_opt[i * n + j]
                = a_par_gpu_img[i * n + j] = num_f;

            b_seq[i * n + j] = b_par_cpu[i * n + j] = b_par_gpu[i * n + j] = b_par_gpu_opt[i * n + j]
                = b_par_gpu_img[i * n + j] = static_cast<float>(10.0) - num_f;
        }
    }
    std::cout << "Generation end" << std::endl;
    std::cout << "*-------------------------* " << std::endl;
    double start = omp_get_wtime();
    SequentialMatrixMultiplication(a_seq, b_seq, c_seq, m, n);
    double end = omp_get_wtime();
    std::cout << "Sequential float time - " << end - start << std::endl;
    start = omp_get_wtime();
    ParallelMatrixMultiplication(a_par_cpu, b_par_cpu, c_par_cpu, m, n);
    end = omp_get_wtime();
    std::cout << "OpenMP float time - " << end - start << std::endl;
    ParallelOpenCLMatrixMultiplication(argv[0], 16, a_par_gpu, b_par_gpu, c_par_gpu, m, n);
    std::cout << "*-------------------------* " << std::endl;
    std::cout << "Float - " << (IsEqual(c_seq, c_par_gpu, n * n) == true ? "true" : "false") << std::endl;

    ParallelOpenCLMatrixMultiplicationOptimized(argv[0], 16, a_par_gpu_opt, b_par_gpu_opt, c_par_gpu_opt, m, n);
    std::cout << "Float optimized - " << (IsEqual(c_seq, c_par_gpu_opt, n * n) == true ? "true" : "false") << std::endl;

    ParallelOpenCLMatrixMultiplicationImage(argv[0], 16, a_par_gpu_img, b_par_gpu_img, c_par_gpu_img, m, n);
    std::cout << "Float image - " << (IsEqual(c_seq, c_par_gpu_img, n * n) == true ? "true" : "false") << std::endl;

    delete[] a_seq;
    delete[] b_seq;
    delete[] c_seq;

    delete[] a_par_cpu;
    delete[] b_par_cpu;
    delete[] c_par_cpu;

    delete[] a_par_gpu;
    delete[] b_par_gpu;
    delete[] c_par_gpu;

    delete[] a_par_gpu_opt;
    delete[] b_par_gpu_opt;
    delete[] c_par_gpu_opt;

    delete[] a_par_gpu_img;
    delete[] b_par_gpu_img;
    delete[] c_par_gpu_img;

    return 0;
}