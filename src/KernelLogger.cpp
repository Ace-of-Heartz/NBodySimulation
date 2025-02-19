//
// Created by ace on 2/8/25.
//

#include "KernelLogger.h"




void KernelLogger::LogMessage(std::string msg)
{
    log_file << msg << std::endl;
}

void KernelLogger::LogKernel(std::string filename, std::string kernel_name, cl_ulong3 global_size, cl_ulong3 workgroup_size)
{
    log_file << filename << std::endl;
    log_file << kernel_name << std::endl;
    log_file << global_size.x << ' ' << global_size.y << ' ' << global_size.z << std::endl;
    log_file << workgroup_size.x << ' ' << workgroup_size.y << ' ' << workgroup_size.z << std::endl;
}

void KernelLogger::CloseLogFile()
{
    log_file.close();
}

void KernelLogger::OpenLogFile(const std::string& filename)
{
    log_file.open(filename);

    if (!log_file.is_open())
    {
        throw std::runtime_error("KernelLogger::OpenLogFile(): failed to open log file");
    }
}

