#include "MyApp.h"

#include <imgui.h>

#include "GLUtils.hpp"

#include <GL/glu.h>
#include <math.h>
#include <random>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/constants.hpp>

#include "oclutils.hpp"



void CMyApp::SetupDebugCallback()
{
	GLint context_flags;
	glGetIntegerv(GL_CONTEXT_FLAGS, &context_flags);
	if (context_flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, GL_FALSE);
		glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR, GL_DONT_CARE, 0, nullptr, GL_FALSE);
		glDebugMessageCallback(SDL_GLDebugMessageCallback, nullptr);
	}
}


bool CMyApp::InitGL()
{
	SetupDebugCallback();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// Create VBO
	vbo = initVBO( num_particles );

	// Create particle shader
	m_program.AttachShader(GL_VERTEX_SHADER, "../shaders/particle.vert");
	m_program.AttachShader(GL_GEOMETRY_SHADER, "../shaders/particle.geom");
	m_program.AttachShader(GL_FRAGMENT_SHADER, "../shaders/particle.frag");

	m_program.BindAttribLoc(0, "vs_in_pos");

	if (!m_program.LinkProgram())
	{
		return false;
	}

	// Load texture
	m_textureID = TextureFromFile("../assets/particle.png");

	return true;
}

bool CMyApp::InitCL()
{  
	try
	{
		///////////////////////////
		// Initialize OpenCL API //
		///////////////////////////

		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		// Try to get the sharing platform!
		bool create_context_success = false;
		for (auto platform : platforms) {
			// Next, create an OpenCL context on the platform.  Attempt to
			// create a GPU-based context.
			cl_context_properties contextProperties[] =
			{
	#ifdef _WIN32
				CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
				CL_GL_CONTEXT_KHR,   (cl_context_properties)wglGetCurrentContext(),
				CL_WGL_HDC_KHR,      (cl_context_properties)wglGetCurrentDC(),
	#elif defined( __GNUC__)
				CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
				CL_GL_CONTEXT_KHR,   (cl_context_properties)glXGetCurrentContext(),
				CL_GLX_DISPLAY_KHR,  (cl_context_properties)glXGetCurrentDisplay(),
	#elif defined(__APPLE__)
				//todo
	#endif
				0
			};
		
			// Create Context
			try {
				context = cl::Context( CL_DEVICE_TYPE_GPU, contextProperties );
				create_context_success = true;
				break;
			} catch (cl::Error error) {}
		}
		
		if(!create_context_success)
			throw cl::Error(CL_INVALID_CONTEXT, "Failed to create CL/GL shared context");

		// Create Command Queue
		cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		std::cout << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
		command_queue = cl::CommandQueue(context, devices[0]);

		/////////////////////////////////
		// Load, then build the kernel //
		/////////////////////////////////

		// Read source file
		std::ifstream sourceFile("../kernels/GLinterop_sol.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

		// Make program of the source code in the context
		program = cl::Program(context, source);
		try {
			program.build(devices);
		} catch (cl::Error error) {
			std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
			throw error;
		}

		// Make kernel
		kernel_update = cl::Kernel(program, "update");
		
		// Create Mem Objs
		cl_vbo_mem = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
		cl_v = cl::Buffer(context, CL_MEM_READ_WRITE, num_particles * sizeof(float) * 2);
		cl_m = cl::Buffer(context, CL_MEM_READ_WRITE, num_particles * sizeof(float));

		///////////////////////////
		// Set-up the simulation //
		///////////////////////////

		/// set masses

		std::vector<float> masses(num_particles, 1);
		masses[rand() % masses.size()] = massiveObjectMass;
		command_queue.enqueueWriteBuffer(cl_m, CL_TRUE, 0, num_particles * sizeof(float), &masses[0]);

		/// set initial velocities

		std::vector<float> vectors(num_particles*2, 0);
		if(bRandVelocities)
		{
			// random velocities
			for (size_t i = 0; i < vectors.size(); i += 2)
			{
				double t = i / double(vectors.size() / 2) * (2 * M_PI);
				double st = sin(t);
				double ct = cos(t);
				double v = 1.7;
				vectors[i + 0] = -ct * v;
				vectors[i + 1] = st * v;
			}
		}

		command_queue.enqueueWriteBuffer(cl_v, CL_TRUE, 0, num_particles * sizeof(float) * 2, &vectors[0]);

		/// set initial positions

		for (size_t i = 0; i < vectors.size(); ++i)
		{
			vectors[i] = ((rand() / float(RAND_MAX)) * 2 - 1);
		}

		if(bRing)
		{
			auto rand_1_1 = []() {
				return (rand() / float(RAND_MAX)) * 2 - 1; 
			};

			// points positioned in a ring
			for (size_t i = 0; i < vectors.size(); i+=2)
			{
				double t = i / double(vectors.size() / 2) * (2 * M_PI);
				double st = sin(t);
				double ct = cos(t);
				double r = 0.25;
				vectors[i + 0] = r * st + rand_1_1() / 5.0;
				vectors[i + 1] = r * ct + rand_1_1() / 5.0;
			}
		}
		
		// positions: upload to GPU global memory using OpenGL
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float* values = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
		for (size_t i = 0; i < vectors.size(); ++i)
			values[i] = vectors[i];
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// kernel args
		kernel_update.setArg(0, cl_v);			// velocities
		kernel_update.setArg(1, cl_vbo_mem);	// positions
		kernel_update.setArg(2, cl_m);			// masses
	}
	catch (cl::Error error)
	{
		// std::cout << error.what() << "(" << oclErrorString(error.err()) << ")" << std::endl;
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "%s", error.what());
		return false;
	}
	return true;
}

bool CMyApp::InitMisc()
{
	start_time = std::chrono::system_clock::now();
	simulation_elapsed_time = 0;
	return true;
}


void CMyApp::CleanCL()
{
	cl_vbo_mem = nullptr;
	cl_v = nullptr;
	cl_m = nullptr;
	program = nullptr;
	kernel_update = nullptr;
	command_queue = nullptr;
}


void CMyApp::Clean()
{

	CleanCL();
	// after we have released the OpenCL references, we can delete the underlying OpenGL objects
	if( vbo != 0 )
	{
		glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
		glDeleteBuffers(1, &vbo);
	}

	glDeleteTextures(1, &m_textureID);
	m_program.Clean();
}

#pragma region Update (CL)

void CMyApp::Update(const SUpdateInfo& update_info)
{
	// static Uint32 last_time = SDL_GetTicks();


	delta_time = update_info.DeltaTimeInSec;
	if (delta_time > 0.05f) delta_time = 0.05f;
	if (delta_time < 0.0001f) delta_time = 0.0001f;
	// if (delta_time > 0.1f) delta_time = 0.1f;

	delta_time *= speed_mult;

	simulation_elapsed_time += delta_time;
	kernel_update.setArg(3, delta_time);

	// CL
	try {
		cl::vector<cl::Memory> acquirable;
		acquirable.push_back(cl_vbo_mem);

		// Acquire GL Objects
		command_queue.enqueueAcquireGLObjects(&acquirable);
		{
			cl::NDRange global(num_particles);

			// interaction & integration
			command_queue.enqueueNDRangeKernel(kernel_update, cl::NullRange, global, cl::NullRange);

			// Wait for all computations to finish
			command_queue.finish();
		}
		// Release GL Objects
		command_queue.enqueueReleaseGLObjects(&acquirable);

	} catch (cl::Error error) {
		// std::cout << error.what() << "(" << oclErrorString(error.err()) << ")" << std::endl;
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,"%s", error.what());
		exit(1);
	}
}

#pragma endregion

#pragma region Render (GL)

void CMyApp::renderVBO( int vbolen )
{
	m_program.On();
	{
		// Shader program parameters
		m_program.SetUniform("particle_size", particle_size);
		m_program.SetTexture("tex0", 0, m_textureID);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(2, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);

		glDrawArrays(GL_POINTS, 0, vbolen);

		glDisableClientState(GL_VERTEX_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	m_program.Off();
}

void CMyApp::Render()
{
	// t�r�lj�k a frampuffert (GL_COLOR_BUFFER_BIT) �s a m�lys�gi Z puffert (GL_DEPTH_BUFFER_BIT)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glDisable(GL_DEPTH_TEST);	
	glDepthMask(GL_FALSE);

	// GL
	renderVBO( num_particles );
}  

void CMyApp::RenderGUI()
{

	static int next_num_particles = num_particles;

	if(ImGui::BeginMainMenuBar())
	{
		if(ImGui::BeginMenu("Information"))
		{

			ImGui::Text("Number of particles: %d", num_particles);
			ImGui::Text("Starting distribution: %s");

			// auto curr = std::chrono::system_clock::now();
			long to_long_milli = simulation_elapsed_time * 60;
			ImGui::Text("Time of simulation: %d:%d:%d" ,
				// std::chrono::duration_cast<std::chrono::minutes>(curr - start_time).count() % 60,
				// std::chrono::duration_cast<std::chrono::seconds>(curr - start_time).count() % 60,
				// std::chrono::duration_cast<std::chrono::milliseconds>(curr - start_time).count() % 60
				(to_long_milli / 60) / (60 ),
				(to_long_milli / 60) % 60,
				to_long_milli % 60
				);

			ImGui::Separator();
			ImGui::Text("Last Frame");
			ImGui::Text("Avg. framerate");


			ImGui::EndMenu();

		}

		if(ImGui::BeginMenu("Controls"))
		{

			ImGui::SliderFloat("Simulation Speed",&speed_mult,0.001,1.5);


			ImGui::Separator();

			if(ImGui::Button("Restart"))
			{
				Clean();
				num_particles = next_num_particles;

				InitGL();
				InitCL();
				InitMisc();
			}

			ImGui::Separator();

			if(ImGui::TreeNode("Configuration"))
			{
				ImGui::InputInt("Particle Count:",&next_num_particles);
				if(ImGui::BeginItemTooltip())
				{
					ImGui::Text("Recommended values: 10.000 - 75.000");
					ImGui::EndTooltip();
				}

				ImGui::SliderFloat("Gravitational Constant",&gravitational_constant,0.00001,1.0f);

				if (ImGui::TreeNode("Particle Distributions"))
				{
					if(ImGui::BeginCombo("Position Distribution",""))
					{
						if(ImGui::Selectable("Uniform"))
						{

						}

						if(ImGui::Selectable("Ring"))
						{

						}

						ImGui::EndCombo();
					}

					if(ImGui::BeginCombo("Velocity Distribution",""))
					{
						if(ImGui::Selectable("Random"))
						{

						}

						if(ImGui::Selectable("Starting outwards"))
						{

						}

						if(ImGui::Selectable("Starting inwards"))
						{

						}

						if(ImGui::Selectable("Functionally Zero"))
						{

						}


						ImGui::EndCombo();
					}

					if (ImGui::BeginCombo("Mass distribution",""))
					{
						if(ImGui::Selectable("Black hole"))
						{

						}
						if(ImGui::Selectable("Uniform"))
						{

						}
						if (ImGui::Selectable("Equal"))
						{

						}
						if (ImGui::Selectable("Random"))
						{

						}
						ImGui::EndCombo();
					}


					ImGui::TreePop();
				}


				ImGui::TreePop();
			}
			ImGui::EndMenu();
		}


		ImGui::EndMainMenuBar();
	}

}




#pragma endregion

#pragma region etc

void CMyApp::KeyboardDown(SDL_KeyboardEvent& key)
{
}

void CMyApp::KeyboardUp(SDL_KeyboardEvent& key)
{
}

void CMyApp::MouseMove(SDL_MouseMotionEvent& mouse)
{
}

void CMyApp::MouseDown(SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseUp(SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseWheel(SDL_MouseWheelEvent& wheel)
{
}

// a k�t param�terbe az �j ablakm�ret sz�less�ge (_w) �s magass�ga (_h) tal�lhat�
void CMyApp::Resize(int _w, int _h)
{
	glViewport(0, 0, _w, _h);
	windowH = _h;
	windowW = _w;
}

void CMyApp::OtherEvent(SDL_Event&)
{

}


CMyApp::CMyApp(void)
{
}

CMyApp::~CMyApp(void)
{
}

#pragma endregion
