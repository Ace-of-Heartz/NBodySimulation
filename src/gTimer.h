#pragma once


#include <GL/glew.h>

class gTimer
{
public:
	gTimer(void);
	~gTimer(void);

	void		Start();
	void		Stop();

	double		StopMillis();

	GLuint64	GetLastDeltaNano();
	double		GetLastDeltaMicro();
	double		GetLastDeltaMilli();
private:
	GLuint		m_queries[2];
	int			m_act;
	GLuint64	m_timer_last_delta;
};

