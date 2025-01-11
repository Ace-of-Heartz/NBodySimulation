// GLEW
#include <GL/glew.h>

// SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include "imgui_impl_sdl2.h"

#include <iostream>
#include <sstream>

#include "MyApp.h"
#include "gTimer.h"

void exitProgram()
{
	SDL_Quit();

	std::cout << "Press any key to exit..." << std::endl;
	std::cin.get();
}

int main( int argc, char* args[] )
{
	// Állítsuk be, hogy kilépés előtt hívja meg a rendszer az exitProgram() függvényt - Kérdés: mi lenne enélkül?
	atexit( exitProgram );



	if ( SDL_Init( SDL_INIT_VIDEO ) == -1 )
	{
		std::cout << "[SDL Init]: Error at SDL initalization" << SDL_GetError() << std::endl;
		return 1;
	}
			
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef _DEBUG
	// ha debug módú a fordítás, legyen az OpenGL context is debug módban, ekkor működik a debug callback
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    //SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE,         32);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE,            8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,          8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,           8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,          8);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,		1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,          24);

	// antialiasing - ha kell
	//SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS,  1);
	//SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES,  4);

	// hozzuk létre az ablakunkat
	SDL_Window *win = nullptr;
    win = SDL_CreateWindow( "N-Body Simulation",		// az ablak fejl�ce
							100,						// az ablak bal-fels� sark�nak kezdeti X koordin�t�ja
							100,						// az ablak bal-fels� sark�nak kezdeti Y koordin�t�ja
							800,						// ablak sz�less�ge
							800,						// �s magass�ga
							SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);			// megjelen�t�si tulajdons�gok

    if (win == nullptr)
	{
		std::cout << "[Creation of Window] Error SDL initialization " << SDL_GetError() << std::endl;
        return 1;
    }

	SDL_GLContext context = SDL_GL_CreateContext(win);
    if (context == nullptr)
	{
		std::cout << "[OGL Context Creation] Error at SDL initialization" << SDL_GetError() << std::endl;
        return 1;
    }	


	SDL_GL_SetSwapInterval(0);


	GLenum error = glewInit();
	if ( error != GLEW_OK )
	{
		std::cout << "[GLEW] Error at GLEW initialization" << std::endl;
		return 1;
	}

	int glVersion[2] = {-1, -1};
	glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]); 
	glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]); 

	SDL_LogInfo(SDL_LOG_CATEGORY_APPLICATION, "Running OpenGL %d.%d", glVersion[0], glVersion[1]);

	if ( glVersion[0] == -1 && glVersion[1] == -1 )
	{
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow( win );

		SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[OGL context creation] Error during the inialization of the OGL context! Maybe one of the SDL_GL_SetAttribute(...) calls is erroneous.");

		return 1;
	}

	std::stringstream window_title;
	window_title << "OpenGL " << glVersion[0] << "." << glVersion[1];
	SDL_SetWindowTitle(win, window_title.str().c_str());

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	ImGui::StyleColorsDark();

	ImGui_ImplSDL2_InitForOpenGL(win, context);
	ImGui_ImplOpenGL3_Init();


	bool quit = false;
	SDL_Event ev;
	
	CMyApp app;
	if (!app.InitGL() || !app.InitCL() || !app.InitMisc())
	{
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow(win);
		SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[app.Init] Error during the initialization of the application!");
		return 1;
	}
	
	gTimer nanoTimer;
	bool ShowImGui = true;


	while (!quit)
	{
		while ( SDL_PollEvent(&ev) )
		{

			ImGui_ImplSDL2_ProcessEvent(&ev);
			bool is_mouse_captured    = ImGui::GetIO().WantCaptureMouse;    //kell-e az imgui-nak az egér
			bool is_keyboard_captured = ImGui::GetIO().WantCaptureKeyboard;	//kell-e az imgui-nak a billentyűzet

			switch (ev.type)
			{
			case SDL_QUIT:
				quit = true;
				break;
			case SDL_KEYDOWN:
				if ( ev.key.keysym.sym == SDLK_ESCAPE )
					quit = true;

				// ALT + ENTER vált teljes képernyőre, és vissza.
				if ( ( ev.key.keysym.sym == SDLK_RETURN )  // Enter le lett nyomva, ...
					 && ( ev.key.keysym.mod & KMOD_ALT )   // az ALTal együtt, ...
					 && !( ev.key.keysym.mod & ( KMOD_SHIFT | KMOD_CTRL | KMOD_GUI ) ) ) // de más modifier gomb nem lett lenyomva.
				{
					Uint32 FullScreenSwitchFlag = ( SDL_GetWindowFlags( win ) & SDL_WINDOW_FULLSCREEN_DESKTOP ) ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP;
					SDL_SetWindowFullscreen( win, FullScreenSwitchFlag );
					is_keyboard_captured = true; // Az ALT+ENTER-t ne kapja meg az alkalmazás.
				}
				// CTRL + F1 ImGui megjelenítése vagy elrejtése
				if ( ( ev.key.keysym.sym == SDLK_F1 )  // F1 le lett nyomva, ...
					 && ( ev.key.keysym.mod & KMOD_CTRL )   // az CTRLal együtt, ...
					 && !( ev.key.keysym.mod & ( KMOD_SHIFT | KMOD_ALT | KMOD_GUI ) ) ) // de más modifier gomb nem lett lenyomva.
				{
					ShowImGui = !ShowImGui;
					is_keyboard_captured = true; // A CTRL+F1-t ne kapja meg az alkalmazás.
				}
				if ( !is_keyboard_captured )
					app.KeyboardDown(ev.key);
				break;
			case SDL_KEYUP:
				if ( !is_keyboard_captured )
					app.KeyboardUp(ev.key);
				break;
			case SDL_MOUSEBUTTONDOWN:
				if ( !is_mouse_captured )
					app.MouseDown(ev.button);
				break;
			case SDL_MOUSEBUTTONUP:
				if ( !is_mouse_captured )
					app.MouseUp(ev.button);
				break;
			case SDL_MOUSEWHEEL:
				if ( !is_mouse_captured )
					app.MouseWheel(ev.wheel);
				break;
			case SDL_MOUSEMOTION:
				if ( !is_mouse_captured )
					app.MouseMove(ev.motion);
				break;
			case SDL_WINDOWEVENT:
				if ( ( ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ) || ( ev.window.event == SDL_WINDOWEVENT_SHOWN ) )
				{
					int w, h;
					SDL_GetWindowSize( win, &w, &h );
					app.Resize( w, h );
				}
				break;
			default:
				app.OtherEvent( ev );
			}

		}


		// Számoljuk ki az update-hez szükséges idő mennyiségeket!
		static Uint32 LastTick = SDL_GetTicks(); // statikusan tároljuk, mi volt az előző "tick".
		Uint32 CurrentTick = SDL_GetTicks(); // Mi az aktuális.
		SUpdateInfo updateInfo // Váltsuk át másodpercekre!
		{
			static_cast<float>(CurrentTick) / 1000.0f,
			static_cast<float>(CurrentTick - LastTick) / 1000.0f
		};
		LastTick = CurrentTick; // Mentsük el utolsóként az aktuális "tick"-et!

		nanoTimer.Start();
		app.Update(updateInfo);
		app.Render();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplSDL2_NewFrame(); //Ezután lehet imgui parancsokat hívni, egészen az ImGui::Render()-ig

		ImGui::NewFrame();
		if ( ShowImGui) app.RenderGUI();
		ImGui::Render();

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		window_title.str(std::string());
		window_title.precision(4);
		window_title << "OpenGL " << glVersion[0] << "." << glVersion[1] << ", last frame: " << nanoTimer.StopMillis() << "ms";
		SDL_SetWindowTitle(win, window_title.str().c_str());

		SDL_GL_SwapWindow(win);
	}




	app.Clean();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow( win );

	return 0;
}