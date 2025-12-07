#version 300 es

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 mv;
uniform mat4 p;

out vec3 fragPosition;
out vec3 fragNormal;

void main() {
  fragPosition = position;
  fragNormal = normalize(normal);
  gl_Position = p * mv * vec4(position, 1.0);
}
