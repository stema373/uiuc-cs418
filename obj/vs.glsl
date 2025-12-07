#version 300 es

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;
layout(location = 3) in vec3 color;

uniform mat4 mv;
uniform mat4 p;

out vec3 fragPosition;
out vec3 fragNormal;
out vec2 fragTexCoord;
out vec3 fragColorAttr;

void main() {
  fragPosition = position;
  fragNormal = normalize(normal);
  gl_Position = p * mv * vec4(position, 1.0);
  fragTexCoord = texcoord;
  fragColorAttr = color;
}
