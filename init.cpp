#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "init.h"


OCLInitialization::OCLInitialization(const std::string& path, const char dim, const size_t* group_size, const size_t* global_size)
    : error_code(0), dimension(dim), group(new size_t[dimension]), gl_size(new size_t[dimension]), kernels{}, buffers{}, images {} {

    for (char i = 0; i < dimension; ++i) {
        group[i] = group_size[i];
        gl_size[i] = global_size[i];
    }

    // Platform:
    cl_uint platformCount;
    if (clGetPlatformIDs(0, nullptr, &platformCount)) {
        std::cerr << "ERROR: GetPlatformIds return error in inicialisation" << std::endl;
        error_code = RED_CODE;
        return;
    }
    if (platformCount) {
        cl_platform_id* platforms = new cl_platform_id[platformCount];
        if (clGetPlatformIDs(platformCount, platforms, nullptr)) {
            std::cerr << "ERROR: GetPlatformIds return error" << std::endl;
            delete[] platforms;
            error_code = RED_CODE;
            return;
        }
        platform = platforms[0];
        delete[] platforms;
    }
    else {
        std::cerr << "ERROR: No OpenCL Platform" << std::endl;
        error_code = RED_CODE;
        return;
    }
    // PlatformInfo();
    // std::cout << "<----------------------->" << std::endl;

    // Context
    cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0 };
    context = clCreateContextFromType((properties) ? properties : nullptr,
        CL_DEVICE_TYPE_GPU, nullptr, nullptr, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating creating context from queue: " << error_code << std::endl;
        error_code = RED_CODE;
        return;
    }
    size_t device_count;
    if (clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &device_count)) {
        std::cerr << "ERROR: No Device in Context" << std::endl;
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }

    if (device_count) {
        cl_device_id* devices = new cl_device_id[device_count];
        if (clGetContextInfo(context, CL_CONTEXT_DEVICES, device_count, devices, nullptr)) {
            std::cerr << "ERROR: GetContextInfo return error" << std::endl;
            clReleaseContext(context);
            delete[] devices;
            error_code = RED_CODE;
            return;
        }
        device = devices[0];
        delete[] devices;
    }
    else {
        std::cerr << "ERROR: No devices" << std::endl;
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }
    // DeviceInfo();
    // std::cout << "<----------------------->" << std::endl;

    // Command Queue
    queue = clCreateCommandQueue(context, device, 0, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating command queue: " << error_code << std::endl;
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }

    // Program
    char* kernel_source{};
    std::shared_ptr<char> ptr(kernel_source);
    size_t kernel_len = GetSourceFromFile(path, &kernel_source);
    const char* ks = kernel_source;
    program = clCreateProgramWithSource(context, 1, &ks, &kernel_len, &error_code);
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating program: " << error_code << std::endl;
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }
    if (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr)) {
        std::cerr << "ERROR: OpenCL Program was not build" << std::endl;
        char log[1024];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 1024, log, nullptr);
        std::cerr << log << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }
}

OCLInitialization::~OCLInitialization() {
    delete[] group;
    delete[] gl_size;
    if (error_code != RED_CODE) {
        for (size_t i = 0; i < buffers.size(); ++i)
            clReleaseMemObject(buffers[i]);
        for (size_t i = 0; i < images.size(); ++i)
            clReleaseMemObject(images[i]);
        for (size_t i = 0; i < kernels.size(); ++i)
            clReleaseKernel(kernels[i]);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        clReleaseDevice(device);
    }
}

cl_kernel& OCLInitialization::GetKernel(const size_t number) {
    return kernels[number];
}

cl_mem& OCLInitialization::GetBuffer(const size_t number) {
    return buffers[number];
}

cl_mem& OCLInitialization::GetImage(const size_t number) {
    return images[number];
}

void OCLInitialization::AddKernel(const std::string& kernel_name) {
    kernels.push_back(clCreateKernel(program, kernel_name.c_str(), &error_code));
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating kernel: " << error_code << std::endl;
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }
}

void OCLInitialization::ExecuteKernel(const cl_kernel& kernel) {
    if (group == 0 && clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, nullptr)) {
        std::cerr << "ERROR: in clGetKernelWorkGroupInfo: " << std::endl;
        for (size_t i = 0; i < buffers.size(); ++i)
            clReleaseMemObject(buffers[i]);
        for (size_t i = 0; i < images.size(); ++i)
            clReleaseMemObject(images[i]);
        for (size_t i = 0; i < kernels.size(); ++i)
            clReleaseKernel(kernels[i]);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }
    if (clEnqueueNDRangeKernel(queue, kernel, dimension, nullptr, gl_size, group, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: cannot execute kernel: " << std::endl;
        for (size_t i = 0; i < buffers.size(); ++i)
            clReleaseMemObject(buffers[i]);
        for (size_t i = 0; i < images.size(); ++i)
            clReleaseMemObject(images[i]);
        for (size_t i = 0; i < kernels.size(); ++i)
            clReleaseKernel(kernels[i]);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }
    if (clFinish(queue)) {
        std::cerr << "ERROR: while finishing all processors: " << std::endl;
        for (size_t i = 0; i < buffers.size(); ++i)
            clReleaseMemObject(buffers[i]);
        for (size_t i = 0; i < images.size(); ++i)
            clReleaseMemObject(images[i]);
        for (size_t i = 0; i < kernels.size(); ++i)
            clReleaseKernel(kernels[i]);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return;
    }
}

size_t OCLInitialization::GetSourceFromFile(const std::string& _path, char** source_code) {
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

void OCLInitialization::PlatformInfo() {
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

void OCLInitialization::DeviceInfo() {
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
              << "Local Memory Size: " << local_mem_size / 1024 << " KB" << std::endl
              << "Global Memory Size: " << global_mem_size / 1048576 << " MB" << std::endl
              << "Max Work Group Total Size: " << work_group_size << std::endl
              << "Max Work-group Dims: ";
    for (size_t i = 0; i < dimention; ++i)
        std::cout << dimention_work_sizes[i] << ", ";
    std::cout << std::endl;
    delete[] dimention_work_sizes;
}
