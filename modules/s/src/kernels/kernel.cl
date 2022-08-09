__kernel void saxpy(const unsigned int n, const float a, __global float* x, const int incx, __global float* y, const int incy) {
    int i = get_global_id(0);

    if (i < 1 + (n-1)*abs(incx))
        y[i * incy] = y[i * incy] + a * x[i * incx];
}

__kernel void daxpy(const unsigned int n, const double a, __global double* x, const int incx, __global double* y, const int incy) {
    int i = get_global_id(0);

    if (i < 1 + (n-1)*abs(incx))
        y[i * incy] = y[i * incy] + a * x[i * incx];
}