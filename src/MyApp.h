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

enum PositionDistr
{
	RING_POS,
	UNIFORM_POS,
};

enum VelocityDistr
{
	RANDOM_VEL,
	STARTING_OUT_VEL,
	STARTING_IN_VEL,
	FUNC_ZERO_VEL
};

enum MassDistr
{
	BLACKHOLE_MASS,
	UNIFORM_MASS,
	EQUAL_MASS,
	RANDOM_MASS
};


class CMyApp
{
public:
	CMyApp(void);
	~CMyApp(void);

	bool InitMisc();
	bool InitGL();
	bool InitCL();

	void Clean();
	void CleanGL();
	void CleanCL();

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

	// GL
	int windowH, windowW;
	GLuint m_vaoID, vbo, texture;
	void renderVBO( int vbolen );

	// CL
	cl::Context context;
	cl::CommandQueue command_queue;
	cl::Program program;

	cl::Kernel kernel_update;

	cl::BufferGL cl_vbo_mem;
	cl::Buffer cl_v, cl_m;

	float delta_time;
	float simulation_elapsed_time = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

	float speed_mult = 1.0;
	int num_particles = 25000;
	float gravitational_constant = 6.67e-11;


	float particle_size = 0.01f;
	const bool bRing = true;
	const bool bRandVelocities = true;

	PositionDistr position_distr;
	VelocityDistr vel_distr;
	MassDistr mass_distr;



	const float massiveObjectMass = 1;

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
		glBufferData(GL_ARRAY_BUFFER, vbolen*sizeof(float)*2, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0); 

		return vbo_buffer;
	}

#pragma endregion

};