#define TILES_SIZE 16
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void MatrixMultiplication(__global float* a, __global float* b, __global float* c, const unsigned int m, const unsigned int n) {
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);

    float res = 0.0;
    for (int i = 0; i < m; ++i)
        res += a[m * tid_y + i] * b[n * i + tid_x];

    c[n * tid_y + tid_x] = res;
}

__kernel void MatrixMultiplicationBlock(__global float* a, __global float* b, __global float* c, const unsigned int m, const unsigned int n) {
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);
    int loc_size_x = get_local_size(0);
    int loc_size_y = get_local_size(1);
    int loc_id_x = get_local_id(0);
    int loc_id_y = get_local_id(1);

    __local float A[TILES_SIZE * TILES_SIZE];
    __local float B[TILES_SIZE * TILES_SIZE];
    float res = 0.0;

    for (int p = 0; p < m / TILES_SIZE; ++p) {
        A[loc_id_y * TILES_SIZE + loc_id_x] = a[tid_y * m + p * loc_size_x + loc_id_x];
        B[loc_id_y * TILES_SIZE + loc_id_x] = b[(p * loc_size_y + loc_id_y) * n + tid_x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILES_SIZE; ++i)
            res += A[TILES_SIZE * loc_id_y + i] * B[TILES_SIZE * i + loc_id_x];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[n * tid_y + tid_x] = res;
}

__kernel void MatrixMultiplicationImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, const unsigned int m, const unsigned int n) {
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);
    int loc_size_x = get_local_size(0);
    int loc_size_y = get_local_size(1);
    int loc_id_x = get_local_id(0);
    int loc_id_y = get_local_id(1);
    int2 grid_id_c = (int2)(tid_x, tid_y);

    __local float A[16 * 16];
    __local float B[16 * 16];
    float res = 0.0;

    for (int p = 0; p < m / TILES_SIZE; ++p) {
        A[loc_id_y * TILES_SIZE + loc_id_x] = read_imagef(a, sampler, (int2)(p * loc_size_x + loc_id_x, tid_y)).x;
        B[loc_id_y * TILES_SIZE + loc_id_x] = read_imagef(b, sampler, (int2)(tid_x, p * loc_size_y + loc_id_y)).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILES_SIZE; ++i)
            res += A[TILES_SIZE * loc_id_y + i] * B[TILES_SIZE * i + loc_id_x];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    write_imagef(c, grid_id_c, (float4)(res, 0.0, 0.0, 0.0));
}