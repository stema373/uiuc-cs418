#version 300 es

layout(location = 0) in vec3 center;
layout(location = 1) in float radius;
layout(location = 2) in vec3 color;

uniform mat4 mv;
uniform mat4 p;
uniform float viewportSize;

out vec3 sphereColor;
out vec3 viewSpaceCenter;
out float sphereRadius;

void main() {
  sphereColor = color;
  sphereRadius = radius;

  vec4 viewPos = mv * vec4(center, 1.0);
  viewSpaceCenter = viewPos.xyz;

  gl_Position = p * viewPos;

  gl_PointSize = viewportSize * p[1][1] * (1.0 / gl_Position.w) * radius;
}
