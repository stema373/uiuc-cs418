#version 300 es
precision highp float;

in float isOutline;
out vec4 color;

void main() {
  vec3 outlineColor = vec3(0.075, 0.16, 0.292);
  vec3 fillColor = vec3(1.0, 0.373, 0.02);
  vec3 c = mix(fillColor, outlineColor, step(0.5, isOutline));
  color = vec4(c, 1.0);
}
