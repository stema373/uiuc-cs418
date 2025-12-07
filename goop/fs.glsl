#version 300 es
precision highp float;

in vec3 sphereColor;
in float sphereRadius;

out vec4 fragColor;

void main() {
  vec2 xy = gl_PointCoord * 2.0 - 1.0;
  float xyLength = length(xy);
  if (xyLength > 1.0) discard;

  fragColor = vec4(sphereColor, 1.0);
}
