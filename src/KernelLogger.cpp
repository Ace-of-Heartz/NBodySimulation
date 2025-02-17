//
// Created by ace on 2/8/25.
//

#include "KernelLogger.h"

// template <class Function>
// void KernelLogger::LogKernel(
//     const std::string msg,
//     const cl::Buffer cl_buffer,
//     const cl::CommandQueue command_queue,
//     const int count, const size_t size,
//     const std::function<std::string(Function)> log_func
//     )
// {
//     log_file << msg;
//
//     std::vector<int> to_log(count,0.0);
//     command_queue.enqueueReadBuffer(cl_buffer,CL_TRUE,0,size * count,&to_log[0]);
//
//     for (int i = 0; i < to_log.size(); i++)
//     {
//         log_file << log_func(i,to_log[i]);
//     }
// }
