#include <CL/cl.h>
#include <string>
#include <vector>

#define RED_CODE 1000

class OCLInitialization {
public:
    OCLInitialization() = delete;
    OCLInitialization(const std::string& path, const char dim, const size_t* group_size, const size_t* global_size);
    OCLInitialization(const OCLInitialization&) = delete;
    OCLInitialization& operator=(const OCLInitialization&) = delete;
    ~OCLInitialization();

    void PlatformInfo();
    void DeviceInfo();
    cl_kernel& GetKernel(const size_t number);
    cl_mem& GetBuffer(const size_t number);
    cl_mem& GetImage(const size_t number);
    void AddKernel(const std::string& kernel_name);
    void ExecuteKernel(const cl_kernel& kernel);

    template <typename T>
    void SetKernelArg(const cl_kernel& kernel, const unsigned int number, const T* memory);

    template <typename T>
    unsigned int AddBuffer(const cl_mem_flags flags, const size_t size);

    template <typename T>
    unsigned int AddImage(const cl_mem_flags flags, const size_t width, const size_t height);

    template <typename T>
    void WriteElementsToBuffer(const unsigned int buffer_number, const size_t size, const T* memory);

    template <typename T>
    void ReadElementsFromBuffer(const unsigned int buffer_number, const size_t size, T* memory);

    template <typename T>
    void WriteElementsToImage(const unsigned int image_number, const size_t width, const size_t height, const T* memory);

    template <typename T>
    void ReadElementsFromImage(const unsigned int image_number, const size_t width, const size_t height, T* memory);

private:
    size_t GetSourceFromFile(const std::string& _path, char** source_code);

    int error_code;
    char dimension;
    size_t* group;
    size_t* gl_size;
    std::vector<cl_kernel> kernels;
    std::vector<cl_mem> buffers;
    std::vector<cl_mem> images;
    cl_platform_id platform;
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
};


template<typename T>
void OCLInitialization::SetKernelArg(const cl_kernel& kernel, const unsigned int number, const T* memory) {
    if (clSetKernelArg(kernel, number, sizeof(T), memory)) {
        std::cerr << "ERROR: cannot load " << number << " argument: " << std::endl;
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
    }
}

template <typename T>
unsigned int OCLInitialization::AddBuffer(const cl_mem_flags flags, const size_t size) {
    buffers.push_back(clCreateBuffer(context, flags, sizeof(T) * size, nullptr, &error_code));
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating input buffer: " << error_code << std::endl;
        for (size_t i = 0; i < buffers.size() - 1; ++i)
            clReleaseMemObject(buffers[i]);
        for (size_t i = 0; i < images.size(); ++i)
            clReleaseMemObject(images[i]);
        for (size_t i = 0; i < kernels.size(); ++i)
            clReleaseKernel(kernels[i]);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return -1;
    }
    return buffers.size();
}

template<typename T>
unsigned int OCLInitialization::AddImage(const cl_mem_flags flags, const size_t width, const size_t height) {
    cl_image_format img_fmt;
    img_fmt.image_channel_data_type = CL_FLOAT;
    img_fmt.image_channel_order = CL_INTENSITY;

    images.push_back(clCreateImage2D(context, flags, &img_fmt, width, height, 0, nullptr, &error_code));
    if (error_code) {
        std::cerr << "ERROR: OpenCL error code in creating input image: " << error_code << std::endl;
        for (size_t i = 0; i < buffers.size(); ++i)
            clReleaseMemObject(buffers[i]);
        for (size_t i = 0; i < images.size() - 1; ++i)
            clReleaseMemObject(images[i]);
        for (size_t i = 0; i < kernels.size(); ++i)
            clReleaseKernel(kernels[i]);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        error_code = RED_CODE;
        return -1;
    }
    return images.size();
}

template<typename T>
void OCLInitialization::WriteElementsToBuffer(const unsigned int buffer_number, const size_t size, const T* memory) {
    if (clEnqueueWriteBuffer(queue, buffers[buffer_number], CL_TRUE, 0, sizeof(T) * size, memory, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: EnqueueWriteBuffer return error: " << std::endl;
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
    }
}

template<typename T>
void OCLInitialization::ReadElementsFromBuffer(const unsigned int buffer_number, const size_t size, T* memory) {
    if (clEnqueueReadBuffer(queue, buffers[buffer_number], CL_TRUE, 0, sizeof(T) * size, memory, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: while rerurning the result: " << std::endl;
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
    }
}

template<typename T>
void OCLInitialization::WriteElementsToImage(const unsigned int image_number,
                                             const size_t width,
                                             const size_t height,
                                             const T* memory) {
    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { width, height, 1 };
    if (clEnqueueWriteImage(queue, images[image_number], CL_TRUE, origin, region, 0, 0, memory, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: EnqueueWriteImage return error: " << std::endl;
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
    }
}

template<typename T>
void OCLInitialization::ReadElementsFromImage(const unsigned int image_number,
                                              const size_t width,
                                              const size_t height,
                                              T* memory) {
    size_t origin[] = { 0, 0, 0 };
    size_t region[] = { width, height, 1 };

    if (clEnqueueReadImage(queue, images[image_number], CL_TRUE, origin, region, 0, 0, memory, 0, nullptr, nullptr)) {
        std::cerr << "ERROR: EnqueueReadImage return error: " << std::endl;
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
    }
}
