#version 300 es

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

uniform mat4 mv;
uniform mat4 p;

out vec3 fragPosition;
out vec3 fragNormal;
out vec2 fragTexCoord;

void main() {
  vec4 worldPos = vec4(position, 1.0);
  gl_Position = p * mv * worldPos;
  fragPosition = worldPos.xyz;
  fragNormal = normalize(normal);
  fragTexCoord = texcoord;
}
