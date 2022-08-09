__kernel void square(__global int* input, __global int* output, const unsigned int count) {
    int b = get_group_id(0);
    int t = get_local_id(0);
    int i = get_global_id(0);

    printf("I am from %d block, %d thread, (global index: %d)", b, t, i);

    if (i < count)
        output[i] = input[i] + i;
}