#pragma once

// GLEW
#include <GL/glew.h>

// SDL
#include <SDL.h>
#include <SDL_opengl.h>

// Utils
#include "gVertexBuffer.h"
#include "gShaderProgram.h"

// CL
#include <iostream>
#include <fstream>
#include <sstream>

#include "Camera.h"
#include "CameraManipulator.h"
#include "SimulationConfig.h"
#include "SimulationUIConfig.h"

//#define CL_HPP_NO_STD_VECTOR
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#define CL_HPP_TARGET_OPENCL_VERSION 210

#ifdef __APPLE__
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#include <CL/cl_gl.h>
#endif

#ifdef __GNUC__
#include <GL/glx.h>
#endif


struct SUpdateInfo
{
	float ElapsedTimeInSec = 0.0f; // Program indulása óta eltelt idő
	float DeltaTimeInSec   = 0.0f; // Előző Update óta eltelt idő
};

class CMyApp
{
public:
	CMyApp();
	~CMyApp();

	void SetupDebugCallback();

	bool InitMisc();
	bool InitGL();
	bool InitCL();

	void InitParticles();

	void Clean();


	void Update(const SUpdateInfo&);
	void Render();
	void RenderGUI();

	void KeyboardDown(SDL_KeyboardEvent&);
	void KeyboardUp(SDL_KeyboardEvent&);
	void MouseMove(SDL_MouseMotionEvent&);
	void MouseDown(SDL_MouseButtonEvent&);
	void MouseUp(SDL_MouseButtonEvent&);
	void MouseWheel(SDL_MouseWheelEvent&);
	void OtherEvent(SDL_Event&);
	void Resize(int, int);
protected:

	bool InitBodyAttributes();
	bool InitObjectAcceleration();
	bool SetKernelArgs();

	void CleanGL();
	void CleanCL();

	std::string filename = "log.txt";
	std::ofstream log_file;
	void LogState();
	void LogChildrenBuffer();
	void LogPositionBuffer();
	void LogVelocityBuffer();
	void LogAccelerationBuffer();
	void LogMassBuffer();
	void LogErrors(std::string);
	void LogDepthBuffer();

	void ResetSimulation();
	// GL
	int windowH, windowW;
	GLuint m_vaoID, vbo, texture;
	void RenderVBO( int vbolen );

	void LoadTexture(const std::string&);

	std::string GetBuildOptions() const;
	void SetKernelConfig();
	void AppendKernelSourceCode(std::string&,const std::string&);

	// Camera
	Camera m_camera;
	CameraManipulator m_cameraManipulator;


	// CL

	int workgroup_size = 16;
	int num_of_nodes;
	int num_of_workgroups;
	int warpsize = 16;
	int max_compute_units;

	bool kernel_debug = true;

	unsigned int max_children;
	unsigned int max_depth = 32;
	int bottom_value = 0;

	bool log_updates = false;
	bool render = true;
	long update_id = 0;


	cl::Context context;
	cl::CommandQueue command_queue;
	cl::Program program;

#pragma region Kernels

	cl::Kernel kernel_init;
	cl::Kernel kernel_update;
	cl::Kernel kernel_update_local;
	cl::Kernel kernel_copy;
	cl::Kernel kernel_hybrid_reduce_root;
	cl::Kernel kernel_parallel_reduce_root;
	cl::Kernel kernel_build_tree; cl::Kernel kernel_build_tree_ext;
	cl::Kernel kernel_saturate_tree; cl::Kernel kernel_saturate_tree_ext;
	cl::Kernel kernel_calculate_force; cl::Kernel kernel_calculate_force_ext;
	cl::Kernel kernel_collision;
	cl::Kernel kernel_sort; cl::Kernel kernel_sort_ext;

#pragma endregion Kernels

#pragma region Buffers

	cl::BufferGL cl_vbo_mem;
	cl::Buffer cl_v, cl_m ,cl_a;

	cl::Buffer cl_temp;

	cl::Buffer cl_p;
	cl::Buffer cl_children;
	cl::Buffer cl_boundary;
	cl::Buffer cl_bottom; // Single variable used for getting the next unused node index
	cl::Buffer cl_errors;
	cl::Buffer cl_bodycount;
	cl::Buffer cl_body_depth_buffer;
	cl::Buffer cl_max_depth;
	cl::Buffer cl_start;
	cl::Buffer cl_sorted;

#pragma endregion Buffers

	float delta_time;
	float simulation_elapsed_time = 0;

	Simulation sim;
	SimulationConfig next_config;

	SimulationUI sim_ui;
	SimulationUIConfig next_ui_config;

#pragma region GL functions

	gShaderProgram	m_program;
	GLuint m_textureID;

	GLuint initVBO(int vbolen )
	{
		GLuint vbo_buffer;
		// generate the buffer
		glGenBuffers(1, &vbo_buffer);

		// bind the buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer);
		glBufferData(GL_ARRAY_BUFFER, vbolen*sizeof(float) * 4, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); 

		return vbo_buffer;
	}

#pragma endregion

};