#include "MyApp.h"
#include "SDL_GLDebugMessageCallback.h"

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

	m_camera.SetView(
	glm::vec3(0.0, 0.0, 9.0),
	glm::vec3(0.0, 0.0, 0.0),
	glm::vec3(0.0, 1.0, 0.0));
	m_cameraManipulator.SetCamera( &m_camera );

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
		} catch (cl::Error& error) {
			std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
			throw error;
		}

		// Make kernel
		kernel_update = cl::Kernel(program, "update");
		
		InitParticles();
	}
	catch (cl::Error& error)
	{
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "%s (%s)", error.what(),oclErrorString(error.err()));
		return false;
	}
	return true;
}

void CMyApp::InitParticles(){
		cl_vbo_mem = cl::BufferGL(context, CL_MEM_WRITE_ONLY, vbo);
		cl_v = cl::Buffer(context, CL_MEM_READ_WRITE, sim.GetConfig().GetNumberOfParticles() * sizeof(float) * 4);
		cl_m = cl::Buffer(context, CL_MEM_READ_WRITE, sim.GetConfig().GetNumberOfParticles() * sizeof(float));

		///////////////////////////
		// Set-up the simulation //
		///////////////////////////

		InitObjectMass();
		InitObjectPositionNVelocity();
		kernel_update.setArg(0, cl_v);
		kernel_update.setArg(1, cl_vbo_mem);
		kernel_update.setArg(2, cl_m);
}

bool CMyApp::InitObjectMass()
{

	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution d(sim.GetConfig().GetMassDistribution().mean,sim.GetConfig().GetMassDistribution().deviation );

	std::vector<float> masses; masses.resize(sim.GetConfig().GetNumberOfParticles(),1.0);


	for (size_t i = 0; i < sim.GetConfig().GetNumberOfParticles(); ++i)
	{
		masses[i] = d(gen);
	}

	for (int i = 0; i < sim.GetConfig().GetNumberOfMassiveObjects(); ++i)
	{
		auto idx = rand() % sim.GetConfig().GetNumberOfMassiveObjects();
		if (masses[idx] == sim.GetConfig().GetMassiveObjectMass())
		{
			--i;
		} else
		{
			masses[idx] = sim.GetConfig().GetMassiveObjectMass();
		}
	}

	const cl_int res = command_queue.enqueueWriteBuffer(cl_m,CL_TRUE,0,sim.GetConfig().GetNumberOfParticles()*sizeof(float),&masses[0]);
	CL_CHECK(res);
	return true;
}



bool CMyApp::InitObjectPositionNVelocity()
{
	std::vector<float> attributes;
	attributes.resize(sim.GetConfig().GetNumberOfParticles() * 4,1.0);

	switch (sim.GetConfig().GetPositionConfig())
	{
	case SPHERE_POS:
		{
			float idx = -static_cast<float>(attributes.size()) / 2.0f;
			const auto N = static_cast<float>(attributes.size());
			for(size_t i = 0; i < attributes.size(); i +=4)
			{
				auto lat = glm::asin( (idx * 2.0f) /(2.0f * N + 1.0f)) * 180.0f / glm::pi<float>();
				//auto lon = 2.0f * glm::pi<float>() * idx / glm::golden_ratio<float>();
				float temp;
				auto lon = glm::modf(idx / glm::golden_ratio<float>(),temp) * 360.0f / glm::golden_ratio<float>();
				// attribute = glm::vec3(r * sin(lat) * sin(lon), r * cos(lon) , r * cos(lat) * sin(lon));

				if (lon < -180.0f) lon += 360.0f;
				else if (lon > 180.0f) lon -= 360.0f;


				attributes[i] = sim.GetConfig().GetStartingVolumeRadius() * sin(lat) * sin(lon);
				attributes[i + 1] = sim.GetConfig().GetStartingVolumeRadius() * cos(lon);
				attributes[i + 2] = sim.GetConfig().GetStartingVolumeRadius() * cos(lat) * sin(lon);
				attributes[i + 3] = 1.0;

				idx += 1.0f;
			}
		}
		break;
	case UNIFORM_POS:
		{
			std::random_device rd{};
			std::mt19937 gen{ rd() };
			std::normal_distribution d(0.0f,sim.GetConfig().GetStartingVolumeRadius());
			for (size_t i = 0; i < attributes.size(); i += 4)
			{
				attributes[i] = d(gen); attributes[i + 1] = d(gen); attributes[i + 2] = d(gen); attributes[i + 3] = 1.0f;
			}
		}
		break;
	}


	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	auto* values = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
	for (size_t i = 0; i < attributes.size(); ++i)
		values[i] = attributes[i];
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	switch (sim.GetConfig().GetVelocityConfig())
	{
	case RANDOM_VEL:
		{

		}
		break;
	case STARTING_OUT_VEL:
		{
			for (size_t i = 0; i < attributes.size(); i += 4)
			{

				auto vel = glm::vec3(attributes[i],attributes[i + 1],attributes[i + 2]) ;
				vel = glm::normalize(vel)  * sim.GetConfig().GetStartingSpeedMul() *  1.0f;
				attributes[i] = vel.x;
				attributes[i + 1] = vel.y;
				attributes[i + 2] = vel.z;
				attributes[i + 3] = 0.0f;
			}
		}
		break;
	case STARTING_IN_VEL:
		{
			for (size_t i = 0; i < attributes.size(); i += 4)
			{
				auto vel = glm::vec3(attributes[i],attributes[i + 1],attributes[i + 2]);
				vel = glm::normalize(vel) * sim.GetConfig().GetStartingSpeedMul()  * -1.0f;
				attributes[i] = vel.x;
				attributes[i + 1] = vel.y;
				attributes[i + 2] = vel.z;
				attributes[i + 3] = 0.0f;
			}
		}
		break;
	case FUNC_ZERO_VEL:
		{
			for (size_t i = 0; i < attributes.size(); i += 4)
			{
				attributes[i] = glm::epsilon<float>();
				attributes[i + 1] = glm::epsilon<float>();
				attributes[i + 2] = glm::epsilon<float>();
				attributes[i + 3] = 0.0f;
			}

		}
		break;
	}

	auto res = command_queue.enqueueWriteBuffer(cl_v, CL_TRUE, 0, sim.GetConfig().GetNumberOfParticles() * sizeof(float) * 4, &attributes[0]);
	CL_CHECK(res);

	return true;
}




bool CMyApp::InitMisc()
{
	simulation_elapsed_time = 0;
	return true;
}

void CMyApp::CleanGL()
{
	m_cameraManipulator.RemoveCamera();

	if( vbo != 0 )
	{
		glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
		glDeleteBuffers(1, &vbo);
	}

	glDeleteTextures(1, &m_textureID);
	m_program.Clean();
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
	CleanGL();
	// after we have released the OpenCL references, we can delete the underlying OpenGL objects

}

#pragma region Update (CL)

void CMyApp::Update(const SUpdateInfo& update_info)
{

	m_cameraManipulator.Update(update_info.DeltaTimeInSec);


	delta_time = update_info.DeltaTimeInSec;
	if (delta_time > 0.005f) delta_time = 0.005f;
	if (delta_time < 0.0001f) delta_time = 0.0001f;
	// if (delta_time > 0.1f) delta_time = 0.1f;

	delta_time *= sim.GetSimulationSpeedMul();

	simulation_elapsed_time += delta_time;
	kernel_update.setArg(3, delta_time);
	kernel_update.setArg(4,sim.GetConfig().GetGravitationalConstant());

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
		SDL_LogError(SDL_LOG_CATEGORY_APPLICATION,"%s (%s)", error.what(),oclErrorString(error.err()) );
		exit(1);
	}
}

#pragma endregion

#pragma region Render (GL)

void CMyApp::RenderVBO( int vbolen )
{
	m_program.On();
	{
		// Shader program parameters
		auto viewProj = m_camera.GetViewProj();
		m_program.SetUniform("particle_size", sim.GetParticleSize());
		m_program.SetUniform("viewProj",viewProj);
		m_program.SetTexture("tex0", 0, m_textureID);


		// glProgramUniformMatrix4fv(m_program)

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(4, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);

		glDrawArrays(GL_POINTS, 0, vbolen);

		glDisableClientState(GL_VERTEX_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	m_program.Off();
}

void CMyApp::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glDisable(GL_DEPTH_TEST);	
	glDepthMask(GL_FALSE);

	// GL
	RenderVBO( sim.GetConfig().GetNumberOfParticles() );
}  

void CMyApp::RenderGUI()
{



	if(ImGui::BeginMainMenuBar())
	{
		if(ImGui::BeginMenu("Information"))
		{

			ImGui::Text("Number of particles: %d", sim.GetConfig().GetNumberOfParticles());
			ImGui::Text("Starting position distribution: %s",sim_ui.GetUIConfig().GetPositionConfigItem());
			ImGui::Text("Starting velocity distribution: %s",sim_ui.GetUIConfig().GetVelocityConfigItem());


			// auto curr = std::chrono::system_clock::now();
			long to_long_milli = simulation_elapsed_time * 60;
			ImGui::Text("Time of simulation: %d:%d:%d" ,

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

			ImGui::SliderFloat("Simulation Speed",&sim.GetSimulationSpeedMul(),0.001,1.5);


			ImGui::Separator();

			if(ImGui::Button("Restart"))
			{

				sim_ui.SetUIConfig(next_ui_config);
				//Clean();
				sim.SetConfig(next_config);

				if(vbo){
					glBindBuffer(GL_ARRAY_BUFFER, vbo);
					glBufferData(GL_ARRAY_BUFFER, sim.GetConfig().GetNumberOfParticles()*sizeof(float) * 4, 0, GL_DYNAMIC_DRAW);
					glBindBuffer(GL_ARRAY_BUFFER, 0);
				}

				InitParticles();

			}

			ImGui::Separator();

			if(ImGui::TreeNode("Configuration"))
			{
				ImGui::InputInt("Particle Count:",&next_config.GetNumberOfParticles());
				if(ImGui::BeginItemTooltip())
				{
					ImGui::Text("Recommended values: 10.000 - 75.000");
					ImGui::EndTooltip();
				}

				// ImGui::SliderFloat("Gravitational Constant Slider",&next_config.GetGravitationalConstant(),1e-11,0.1f,"%.11f");
				ImGui::InputFloat("Gravitational Constant",&next_config.GetGravitationalConstant(),1e-6,1e-3,"%.11f");

				if (ImGui::TreeNode("Particle Distributions & Settings"))
				{

					if(ImGui::TreeNode("Position"))
					{

						if(ImGui::BeginCombo("Position Distribution",next_ui_config.GetPositionConfigItem()))
						{

							for (auto& [name,value] : SimulationUI::pos_config_items)
							{
								if (ImGui::Selectable(name,false))
								{
									next_config.SetPositionConfig(value);
									next_ui_config.SetPositionConfigItem(name);
								}
							}


							ImGui::EndCombo();
						}


						ImGui::SliderFloat("Starting Volume Radius:",&next_config.GetStartingVolumeRadius(),0.0f,10.0f);
						ImGui::TreePop();
					}



					if (ImGui::TreeNode("Velocity"))
					{
						if(ImGui::BeginCombo("Velocity Distribution",next_ui_config.GetVelocityConfigItem()))
						{
							for (auto& [name,value] : SimulationUI::vel_config_items)
							{
								if (ImGui::Selectable(name))
								{
									next_config.SetVelocityConfig(value);
									next_ui_config.SetVelocityConfigItem(name);
								}
							}

							ImGui::EndCombo();
						}
						ImGui::TreePop();
					}

					if(ImGui::TreeNode("Mass"))
					{
						ImGui::SliderFloat("Mass Distribution Mean",&next_config.GetMassDistribution().mean,0.0f,100.0f);
						ImGui::SliderFloat("Mass Distribution Deviation",&next_config.GetMassDistribution().deviation,0,next_config.GetMassDistribution().mean);
						if (next_config.GetMassDistribution().deviation > next_config.GetMassDistribution().mean) next_config.GetMassDistribution().deviation = next_config.GetMassDistribution().mean;

						ImGui::InputInt("Number of Massive Particles:",&next_config.GetNumberOfMassiveObjects(),1,10);
						if (next_config.GetNumberOfMassiveObjects() > next_config.GetNumberOfParticles()) next_config.GetNumberOfParticles() = next_config.GetNumberOfMassiveObjects();
						ImGui::SliderFloat("Mass of Massive Particles:",&next_config.GetMassiveObjectMass(),0.0f,100.0f);
						ImGui::TreePop();
					}

					ImGui::TreePop();

				}



				ImGui::TreePop();
			}
			ImGui::EndMenu();
		}

		if(ImGui::BeginMenu("View"))
		{
			ImGui::SliderFloat("Particle Size:",&sim.GetParticleSize(),0.05f,1.0f);
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}

}




#pragma endregion

#pragma region etc

void CMyApp::KeyboardDown(SDL_KeyboardEvent& key)
{
	m_cameraManipulator.KeyboardDown( key );
}

void CMyApp::KeyboardUp(SDL_KeyboardEvent& key)
{
	m_cameraManipulator.KeyboardUp( key );
}

void CMyApp::MouseMove(SDL_MouseMotionEvent& mouse)
{
	m_cameraManipulator.MouseMove( mouse );
}

void CMyApp::MouseDown(SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseUp(SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseWheel(SDL_MouseWheelEvent& wheel)
{

	m_cameraManipulator.MouseWheel(wheel);
}

// a k�t param�terbe az �j ablakm�ret sz�less�ge (_w) �s magass�ga (_h) tal�lhat�
void CMyApp::Resize(int _w, int _h)
{
	glViewport(0, 0, _w, _h);
	windowH = _h;
	windowW = _w;

	m_camera.SetAspect(static_cast<float>(_w) / _h);
}

void CMyApp::OtherEvent(SDL_Event& ev)
{
	if ( ev.type == SDL_DROPFILE || ev.type == SDL_DROPTEXT) {

		std::string filename = std::string(ev.drop.file);

		if (filename.rfind(".png") != std::string::npos) {
			LoadTexture(filename);
		}
		SDL_free(ev.drop.file);
	}
}

CMyApp::CMyApp(void) : sim(), next_config(), sim_ui(), next_ui_config()
{
}

CMyApp::~CMyApp(void)
{
}

#pragma endregion
