#version 150

uniform sampler2D tex0;

in vec2 Vertex_UV;
in vec4 Vertex_Color;
out vec4 FragColor;

void main(void)
{
	vec2 uv = Vertex_UV.xy;
	uv.y *= -1.0;
	//FragColor = vec4(texture(tex0, uv).rgb, 1.0) * Vertex_Color;
	FragColor = texture(tex0, uv) * Vertex_Color * vec4(1,1,1,0.5);
	//FragColor = vec4(texture(tex0, uv).rgb, 1.5) * Vertex_Color;
}