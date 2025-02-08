# N-Body Simulation

## Description

**N-Body simulation** implemented in 3D with OpenCL 2.0 and OpenGL. Based on [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf) and [GPU-N-Body Repository](https://github.com/bneukom/gpu-nbody), see below for details.
Currently supports 3 types of algorithms:
- Brute Force using only Global Memory
- Brute Force using Locally Cached Memory
- Barnes-Hut Algorithm using Octrees

## Examples
### Particles randomized with normal distribution
[NormalDistConfigurration.webm](assets/NormalDistConfigurration.webm)

### Sphere configuration with particles
[SphereConfiguration.webm](assets/SphereConfiguration.webm)

## APIs
- OpenGL3
- ImGui
- SDL2



## Credits & Sources

- The implementation is mainly based on [An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm](https://iss.oden.utexas.edu/Publications/Papers/burtscher11.pdf) paper, discussing the considerations and optimizations used for efficiently implementing the Barnes Hut algorithm used for solvign the N-Body problem.
- The implementation is also based on the [GPU-N-Body Repository](https://github.com/bneukom/gpu-nbody) done by Benjamin Neukom. Although the kernels were aimed to be refined and customized more for my uses, the main ideas and approaches behind them remain the same from their implementation. It also proved to be a great resource for better understanding memory and visibility management with atomic variables, and showing how general idea behind parallel octree handling 
- The locally optimized brute force solution for the N-Body problem is based on [Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda).
- A great resource discussing Fibonacci lattices and the underlying formulas: [Fibonacci Lattices](https://observablehq.com/@meetamit/fibonacci-lattices)
For a better understanding over the **N-Body problem** and the **Barnes Hut algorithm**, the following resources were used:
- [The Barnes-Hut Algorithm](https://arborjs.org/docs/barnes-hut) (HTTP only site)

## Known bugs
- The Barnes Hut depth limit seems to be unreasonably limiting at the moment, which might indicate that there is other problem behind it.

## Future Plans

- Seek other ways to increase limit for Barnes Hut tree depth
    - Spatial Hashing
- Implement proper logging features:
  - Log starting values (for later review) + Implement config load in using log files
  - Log kernel inputs and outputs
- Implement other N-Body problem solver algorithms on CPU and GPU
- Implement better interface for using different types of N-Body algorithms
- Work and improve graphics calls
  - Vulkan API (in the near future sometime)
  - More, custom shaders:
    - Particle size depending on weight
    - Particle colour depending on weight/acceleration/speed
    - Bounding box draws
