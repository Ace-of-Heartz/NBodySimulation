#version 150

in vec4 vs_in_pos;
 
out Vertex
{
	vec4 color;
} vertex;

//uniform mat4 world;
//uniform mat4 worldIT;
uniform mat4 viewProj;

void main()
{

	gl_Position = viewProj * vs_in_pos;
	vertex.color = vec4(1,1,1,1);
}