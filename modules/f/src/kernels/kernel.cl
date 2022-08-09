__kernel void Jacobi(__global float* A, __global float* b, __global float* x, __global float* x_current,
                     const unsigned int size, const float eps, const int iter, __global int* iter_count) {
    int tid_x = get_global_id(0);
    float bi = b[tid_x];
    float aii = A[tid_x * size + tid_x];
    do {
        barrier(CLK_GLOBAL_MEM_FENCE);
        float sum = 0.0;
        for (size_t j = 0; j < size; ++j)
            if (j != tid_x)
                sum += x_current[j] * A[tid_x * size + j];
        x[tid_x] = x_current[tid_x];
        x_current[tid_x] = (bi - sum) / aii;
        if (tid_x == 0)
            ++(*iter_count);
        barrier(CLK_GLOBAL_MEM_FENCE);
    } while (*iter_count < iter && fabs(x_current[0] - x[0]) / fabs(x[0]) > eps);
    if (tid_x == 0 && *iter_count >= iter)
        printf("iter count - %d\n", *iter_count);
}