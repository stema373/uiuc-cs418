#version 300 es
precision highp float;

in float outline;
out vec4 color;

void main() {
  vec3 outlineColor = vec3(0.075, 0.16, 0.292);
  vec3 fillColor = vec3(1.0, 0.373, 0.02);
  color = outline == 1.0 ? vec4(outlineColor, 1.0) : vec4(fillColor, 1.0);
}
