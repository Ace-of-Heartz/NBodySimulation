//
// Created by ace on 2/8/25.
//

#ifndef KERNELLOGGER_H
#define KERNELLOGGER_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#define CL_HPP_TARGET_OPENCL_VERSION 210

#include <CL/opencl.hpp>
#include <string>
#include <fstream>

class KernelLogger {
public:
    template <class ELEM>
    static void LogKernelParam(
        cl::Buffer cl_buffer,
        cl::CommandQueue command_queue,
        int item_count,
        int per_item,
        size_t size,
        std::function<std::string(int,int,ELEM*)> log_func)
    {
        log_file << "<size=" << item_count * per_item << '>' << std::endl;
        std::vector<ELEM> to_log(item_count * per_item,0.0);
        command_queue.enqueueReadBuffer(cl_buffer,CL_TRUE,0,size * per_item * item_count,&to_log[0]);

        for (int i = 0; i < item_count; i++)
        {
            log_file << log_func(i,per_item,to_log.data());
        }
        log_file << std::endl;
    }

    template <typename T>
    static void LogKernelParam(T param)
    {
        log_file << "<size=" << 1 << '>' << std::endl;
        log_file << param << std::endl;
    }



    static void LogMessage(
        std::string msg
    );

    static void LogKernel(std::string filename, std::string kernel_name, cl_ulong3 global_size, cl_ulong3 workgroup_size);


    static void OpenLogFile(const std::string& filename);
    static void CloseLogFile();

private:
    static inline std::ofstream log_file;

};

template <int N>
struct KernelAbstraction
{
    int kernel_arg_count;
    std::string filename;
    std::string kernel_name;
    cl::NDRange global_size;
    cl::NDRange local_size;

    cl::Buffer* buffers[N];
    int buffer_sizes[N];
    int elem_sizes[N];
};

#endif //KERNELLOGGER_H
