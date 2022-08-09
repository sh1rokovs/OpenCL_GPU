#include <CL/cl.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <random>
#include "../../../init.h"

void SequentialSaxpy(const unsigned int n, const float a, float* x, const int incx, float* y, const int incy) {
    for (int i = 0; i < n; ++i)
        y[i * incy] = y[i * incy] + a * x[i * incx];
}

void ParallelOpenMPSaxpy(const unsigned int n, const float a, float* x, const int incx, float* y, const int incy) {
    int num_tr = omp_get_max_threads();
    //std::cout << num_tr << std::endl;
#pragma omp parallel num_threads(num_tr)
    {
#pragma omp for schedule(static, n / num_tr)
        for (int i = 0; i < n; ++i) {
            y[i * incy] = y[i * incy] + a * x[i * incx];
            //std::cout << y[i * incy] << " - " << omp_get_thread_num() << std::endl;
        }
    }
}

void ParallelOpenCLSaxpy(const char* path, const size_t group, const unsigned int n, const float a,
    float* x, const int incx, const size_t size_x, float* y, const int incy, const size_t size_y) {

    size_t group_size[] = { group };
    size_t size[] = { n };
    OCLInitialization pr(path, 1, group_size, size);
    pr.AddKernel("saxpy");
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, size_y);
    pr.AddBuffer<float>(CL_MEM_READ_ONLY, size_x);
    pr.WriteElementsToBuffer(0, size_y, y);
    pr.WriteElementsToBuffer(1, size_x, x);
    pr.SetKernelArg(pr.GetKernel(0), 0, &n);
    pr.SetKernelArg(pr.GetKernel(0), 1, &a);
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 3, &incx);
    pr.SetKernelArg(pr.GetKernel(0), 4, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 5, &incy);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU float time - " << end - start << std::endl;
    pr.ReadElementsFromBuffer(0, size_y, y);
}

void SequentialDaxpy(const unsigned int n, const double a, double* x, const int incx, double* y, const int incy) {
    for (int i = 0; i < n; ++i)
        y[i * incy] = y[i * incy] + a * x[i * incx];
}

void ParallelOpenMPDaxpy(const unsigned int n, const double a, double* x, const int incx, double* y, const int incy) {
    int num_tr = omp_get_max_threads();
    //std::cout << num_tr << std::endl;
#pragma omp parallel num_threads(num_tr)
    {
#pragma omp for schedule(static, n / num_tr)
        for (int i = 0; i < n; ++i) {
            y[i * incy] = y[i * incy] + a * x[i * incx];
            //std::cout << y[i * incy] << " - " << omp_get_thread_num() << std::endl;
        }
    }
}

void ParallelOpenCLDaxpy(const char* path, const size_t group, const unsigned int n, const double a,
    double* x, const int incx, const size_t size_x, double* y, const int incy, const size_t size_y) {

    size_t group_size[] = { group };
    size_t size[] = { n };
    OCLInitialization pr(path, 1, group_size, size);
    pr.AddKernel("daxpy");
    pr.AddBuffer<double>(CL_MEM_READ_WRITE, size_y);
    pr.AddBuffer<double>(CL_MEM_READ_ONLY, size_x);
    pr.WriteElementsToBuffer(0, size_y, y);
    pr.WriteElementsToBuffer(1, size_x, x);
    pr.SetKernelArg(pr.GetKernel(0), 0, &n);
    pr.SetKernelArg(pr.GetKernel(0), 1, &a);
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 3, &incx);
    pr.SetKernelArg(pr.GetKernel(0), 4, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 5, &incy);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU float time - " << end - start << std::endl;
    pr.ReadElementsFromBuffer(0, size_y, y);
}

template <typename T>
bool IsEqual(const T* a, const T* b, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        //printf("%d - %lf = %lf\n", i, a[i], b[i]);
        if (std::abs(a[i] - b[i]) > 0.000001)
            return false;
    }
    return true;
}

int main(int argc, char** argv) {
    const unsigned int size = 100000000;

    // init float
    float* y_seq = new float[size];
    float* x_seq = new float[size];

    float* y_par_cpu = new float[size];
    float* x_par_cpu = new float[size];

    float* y_par_gpu = new float[size];
    float* x_par_gpu = new float[size];

    // init double
    double* y_seq_d = new double[size];
    double* x_seq_d = new double[size];

    double* y_par_cpu_d = new double[size];
    double* x_par_cpu_d = new double[size];

    double* y_par_gpu_d = new double[size];
    double* x_par_gpu_d = new double[size];

    float f = 2.0;
    double d = 2.0;
    double lower_bound = 0;
    double upper_bound = 1000;
    std::cout << "gen start" << std::endl;
    std::uniform_real_distribution<float> unif_f(lower_bound, upper_bound);
    std::uniform_real_distribution<double> unif_d(lower_bound, upper_bound);
    std::default_random_engine re;
    for (int i = 0; i < size; ++i) {
        float num_f = unif_f(re);
        double num_d = unif_d(re);

        y_seq[i] = y_par_cpu[i] = y_par_gpu[i] = num_f;
        y_seq_d[i] = y_par_cpu_d[i] = y_par_gpu_d[i] = num_d;

        x_seq[i] = x_par_cpu[i] = x_par_gpu[i] = static_cast<float>(size - 1) - num_f;
        x_seq_d[i] = x_par_cpu_d[i] = x_par_gpu_d[i] = static_cast<double>(size - 1) - num_d;
    }
    std::cout << "gen end" << std::endl;
    int incx = 1, incy = 1;

    double start = omp_get_wtime();
    SequentialSaxpy(size, f, x_seq, incx, y_seq, incy);
    double end = omp_get_wtime();
    std::cout << "Sequential float time - " << end - start << std::endl;
    start = omp_get_wtime();
    ParallelOpenMPSaxpy(size, f, x_par_cpu, incx, y_par_cpu, incy);
    end = omp_get_wtime();
    std::cout << "OpenMP float time - " << end - start << std::endl;
    //start = omp_get_wtime();
    ParallelOpenCLSaxpy(argv[0], 256, size, f, x_par_gpu, incx, size, y_par_gpu, incy, size);
    //end = omp_get_wtime();
    //std::cout << "GPU float time - " << end - start << std::endl;

    std::cout << "Float compare - " << (IsEqual(y_seq, y_par_gpu, size) == true ? "true" : "false") << std::endl;

    start = omp_get_wtime();
    SequentialDaxpy(size, d, x_seq_d, incx, y_seq_d, incy);
    end = omp_get_wtime();
    std::cout << "Sequential double time - " << end - start << std::endl;
    start = omp_get_wtime();
    ParallelOpenMPDaxpy(size, d, x_par_cpu_d, incx, y_par_cpu_d, incy);
    end = omp_get_wtime();
    std::cout << "OpenMP double time - " << end - start << std::endl;
    //start = omp_get_wtime();
    ParallelOpenCLDaxpy(argv[0], 256, size, d, x_par_gpu_d, incx, size, y_par_gpu_d, incy, size);
    //end = omp_get_wtime();
    //std::cout << "GPU float time - " << end - start << std::endl;
    std::cout << "Double compare - " << (IsEqual(y_seq, y_par_gpu, size) == true ? "true" : "false") << std::endl;

    // delete float
    delete[] y_seq;
    delete[] x_seq;

    delete[] y_par_cpu;
    delete[] x_par_cpu;

    delete[] y_par_gpu;
    delete[] x_par_gpu;

    // delete double
    delete[] y_seq_d;
    delete[] x_seq_d;

    delete[] y_par_cpu_d;
    delete[] x_par_cpu_d;

    delete[] y_par_gpu_d;
    delete[] x_par_gpu_d;

    return 0;
}