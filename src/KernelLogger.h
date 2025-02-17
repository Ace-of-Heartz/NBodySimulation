//
// Created by ace on 2/8/25.
//

#ifndef KERNELLOGGER_H
#define KERNELLOGGER_H

// //#define CL_HPP_NO_STD_VECTOR
// #define CL_HPP_ENABLE_EXCEPTIONS
// #define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
// #define CL_TARGET_OPENCL_VERSION 210
// #define CL_HPP_TARGET_OPENCL_VERSION 210

#include <CL/opencl.hpp>
#include <string>
#include <fstream>

class KernelLogger {
public:
    template <class Function>
    static void LogKernel(
        std::string msg,
        cl::Buffer cl_buffer,
        cl::CommandQueue command_queue,
        int count,
        size_t size,
        std::function<std::string(Function)> log_func);

    static void OpenLogFile();
    static void CloseLogFile();

private:
    static inline std::ofstream log_file;
    static inline const std::string log_file_name;
};



#endif //KERNELLOGGER_H
