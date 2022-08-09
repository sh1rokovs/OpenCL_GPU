#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

size_t GetSourceFromFile(const std::string& _path, char** source_code) {
    size_t number = _path.find_last_of("/\\");
    std::string path = _path.substr(0, number);
    std::ifstream file(path + "/kernels/kernel.cl", std::ios_base::in | std::ios_base::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: opening resource file /kernels/kernel.cl is failed!" << std::endl;
    }
    std::stringstream source;
    source << file.rdbuf();
    file.close();
    std::string str = source.str();
    (*source_code) = new char[str.size()];
    for (size_t i = 0; i < str.size(); ++i)
        (*source_code)[i] = str[i];
    return str.size();
}

void PlatformInfo(const cl_platform_id platform) {
    char platformName[128];
    char platformVendor[128];
    char platformVersion[128];
    if (clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 128, platformVersion, nullptr)) {
        std::cerr << "ERROR: GetPlatformInfo return error (CL PLATFORM VERSION)" << std::endl;
        return;
    }
    if (clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 128, platformVendor, nullptr)) {
        std::cerr << "ERROR: GetPlatformInfo return error (PLATFORM VENDOR)" << std::endl;
        return;
    }
    if (clGetPlatformInfo(platform, CL_PLATFORM_NAME, 128, platformName, nullptr)) {
        std::cerr << "ERROR: GetPlatformInfo return error (PLATFORM NAME)" << std::endl;
        return;
    }
    std::cout << platformName << std::endl << platformVendor << std::endl << platformVersion << std::endl;
}

void DeviceInfo(const cl_device_id device) {
    char device_name[128];
    char device_ocl_vercion[128];
    cl_uint compute_units;
    cl_ulong local_mem_size;
    cl_ulong global_mem_size;
    size_t work_group_size;
    cl_uint dimention;

    if (clGetDeviceInfo(device, CL_DEVICE_NAME, 128, &device_name, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (NAME)" << std::endl;
        return;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 128, &device_ocl_vercion, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (CL VERSION)" << std::endl;
        return;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (COMPUTE UNITS)" << std::endl;
        return;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (LOCAK MEM)" << std::endl;
        return;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (GLOBAL MEM)" << std::endl;
        return;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (WORK GROUP SIZE)" << std::endl;
        return;
    }
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimention, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (ITEM DIMENSIONS)" << std::endl;
        return;
    }
    size_t* dimention_work_sizes = new size_t[dimention];
    if (clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, dimention * sizeof(size_t), dimention_work_sizes, nullptr)) {
        std::cerr << "ERROR: GetDeviceInfo return error (ITEM SIZES)" << std::endl;
        return;
    }

    std::cout << "Name: " << device_name << std::endl
              << "Device OpenCL Version: " << device_ocl_vercion << std::endl
              << "Maximun Compute Units: " << compute_units << std::endl
              << "Local Memory Size: " << local_mem_size / 1024 << " KB" <<std::endl
              << "Global Memory Size: " << global_mem_size / 1048576 << " MB" << std::endl
              << "Max Work Group Total Size: " << work_group_size << std::endl
              << "Max Work-group Dims: ";
    for (size_t i = 0; i < dimention; ++i)
        std::cout << dimention_work_sizes[i] << ", ";
    std::cout << std::endl;
    delete[] dimention_work_sizes;
}

int main(int argc, char** argv) {

    // Platform:
    cl_uint platformCount = 0;
    if (clGetPlatformIDs(0, nullptr, &platformCount)) {
        std::cerr << "ERROR: GetPlatformIds return error in inicialisation" << std::endl;
        return -1;
    }
    cl_platform_id platform;
    if (platformCount) {
        cl_platform_id* platforms = new cl_platform_id[platformCount];
        if (clGetPlatformIDs(platformCount, platforms, nullptr)) {
            std::cerr << "ERROR: GetPlatformIds return error" << std::endl;
            delete[] platforms;
            return -1;
        }
        platform = platforms[0];
        delete[] platforms;
    } else {
        std::cerr << "ERROR: No OpenCL Platform" << std::endl;
        return -1;
    }
    PlatformInfo(platform);
    std::cout << "<----------------------->" << std::endl;

    // Context
    int error_code;
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0 };
    cl_context context = clCreateContextFromType((properties) ? properties : nullptr,
        CL_DEVICE_TYPE_GPU, nullptr, nullptr, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating creating context from queue: " << error_code << std::endl;
    }
    size_t device_count;
    if (clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &device_count)) {
        std::cerr << "ERROR: No Device in Context" << std::endl;
        clReleaseContext(context);
        return -1;
    }
    
    cl_device_id device;
    if (device_count) {
        cl_device_id* devices = new cl_device_id[device_count];
        if (clGetContextInfo(context, CL_CONTEXT_DEVICES, device_count, devices, nullptr)) {
            std::cerr << "ERROR: GetContextInfo return error" << std::endl;
            clReleaseContext(context);
            delete[] devices;
            return -1;
        }
        device = devices[0];
        delete[] devices;
    } else {
        std::cerr << "ERROR: No devices" << std::endl;
        clReleaseContext(context);
        return -1;
    }
    DeviceInfo(device);
    std::cout << "<----------------------->" << std::endl;

    // Command Queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating command queue: " << error_code << std::endl;
        clReleaseContext(context);
        return -1;
    }

    // Kernel and Program
    char* kernel_source{};
    std::shared_ptr<char> ptr(kernel_source);
    size_t kernel_len = GetSourceFromFile(argv[0], &kernel_source);
    const char* ks = kernel_source;
    cl_program program = clCreateProgramWithSource(context, 1, &ks, &kernel_len, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating program: " << error_code << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    if (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr)) {
        std::cerr << "ERROR: OpenCL Program was not build" << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    cl_kernel _kernel = clCreateKernel(program, "square", &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating kernel: " << error_code << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }

    // input and output values
    const size_t size = 1024;
    int input[size];
    int output[size];
    for (int i = 0; i < size; ++i) {
        input[i] = 0;
    }
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, nullptr, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating input buffer: " << error_code << std::endl;
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, nullptr, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating output buffer: " << error_code << std::endl;
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    if (clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(float) * size, input, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: EnqueueWriteBuffer return error: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }

    // Starting kernel
    if (clSetKernelArg(_kernel, 0, sizeof(cl_mem), &input_buffer)) {
        std::cerr << "ERROR: cannot load 1-st argument: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    if (clSetKernelArg(_kernel, 1, sizeof(cl_mem), &output_buffer)) {
        std::cerr << "ERROR: cannot load 2-nd argument: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    if (clSetKernelArg(_kernel, 2, sizeof(unsigned int), &size)) {
        std::cerr << "ERROR: cannot load 3-rd argument: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    size_t group;
    if (clGetKernelWorkGroupInfo(_kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr)) {
        std::cerr << "ERROR: in clGetKernelWorkGroupInfo: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
     
    if (clEnqueueNDRangeKernel(queue, _kernel, 1, nullptr, &size, &group, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: cannot execute kernel: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }
    if (clFinish(queue)) {
        std::cerr << "ERROR: while finishing all processors: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }

    // Coping from kernel
    if (clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(int) * size, output, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: while rerurning the result: " << std::endl;
        clReleaseMemObject(output_buffer);
        clReleaseMemObject(input_buffer);
        clReleaseKernel(_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return -1;
    }

    // Cleaning memory
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseProgram(program);
    clReleaseKernel(_kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    for (size_t i = 0; i < size; ++i)
        std::cout << output[i] << ", ";
    std::cout << std::endl;

    return 0;
}