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
	SPHERE_POS,
	UNIFORM_POS,
};

enum VelocityDistr
{
	RANDOM_VEL,
	STARTING_OUT_VEL,
	STARTING_IN_VEL,
	FUNC_ZERO_VEL
};

struct NormalDistribution
{
	float mean;
	float deviation;
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

	bool InitObjectMass();
	bool InitObjectPositionNVelocity();

	void CleanGL();
	void CleanCL();
	// GL
	int windowH, windowW;
	GLuint m_vaoID, vbo, texture;
	void RenderVBO( int vbolen );

	// Camera
	Camera m_camera;
	CameraManipulator m_cameraManipulator;



	// CL
	cl::Context context;
	cl::CommandQueue command_queue;
	cl::Program program;

	cl::Kernel kernel_update;

	cl::BufferGL cl_vbo_mem;
	cl::Buffer cl_v, cl_m;

	float delta_time;
	float simulation_elapsed_time = 0;

	float speed_mult = 1.0;
	int num_particles = 25000;
	float gravitational_constant = 6.67e-11;


	float particle_size = 0.05f;


	PositionDistr position_distr = SPHERE_POS;
	VelocityDistr vel_distr = FUNC_ZERO_VEL;
	NormalDistribution mass_normal = NormalDistribution(0.5,0.2);


	int num_massive_particles = 3;
	float massive_particle_mass = 2;
	float starting_velocity = 1.0f;
	float starting_volume_radius = 0.5f;


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