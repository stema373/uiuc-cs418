#version 300 es
precision highp float;

uniform vec3 uColor;
out vec4 color;

void main() {
  color = vec4(uColor, 1.0);
}
