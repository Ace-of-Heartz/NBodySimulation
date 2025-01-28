# N-Body Simulation

## Description

**N-Body simulation** implemented in 3D with OpenCL 2.0 and OpenGL.
Currently supports 3 types of algorithms:
- Brute Force using only Global Memory
- Brute Force using Locally Cached Memory
- Barnes-Hut Algorithm using Octrees

## APIs
- OpenGL3
- ImGui
- SDL2


## Credits & Sources

- The implementation is mainly based on [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf) paper, discussing the considerations and optimizations used for efficiently implementing the Barnes Hut algorithm used for solvign the N-Body problem.
- The implementation is also based on the [GPU-N-Body Repository](https://github.com/bneukom/gpu-nbody) done by Benjamin Neukom. Although the kernels were aimed to be refined and customized more for my uses, the main ideas and approaches behind them remain the same from their implementation.
- The locally optimized brute force solution for the N-Body problem is based on [Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda).

For a better understanding over the **N-Body problem** and the **Barnes Hut algorithm**, the following resources were used:
- [The Barnes-Hut Algorithm](https://arborjs.org/docs/barnes-hut) (HTTP only site)