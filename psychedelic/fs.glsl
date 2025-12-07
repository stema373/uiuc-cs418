#version 300 es
precision highp float;
in vec4 pos;
out vec4 color;
uniform float seconds;
void main() {
  float t = seconds * 0.5;

  float poly1 = pos.x * pos.x - pos.y * pos.y;
  float poly2 = pos.x * pos.y * (pos.x + pos.y);

  float wave1 = sin(pos.x * 3.0 + t) + cos(pos.y * 2.5 - t * 1.2) + poly1 * 0.5;
  float wave2 = sin(pos.y * 4.0 - t * 0.7) + cos(pos.x * 2.0 + t) + poly2 * 0.4;

  float wave3 = sin(wave1 * wave2 + t * 0.8);
  float poly3 = poly1 * poly2;

  float r = sin(wave1 + wave3 + t * 0.6);
  float g = sin(wave2 - wave3 - t * 1.1);
  float b = sin(poly3 + wave3 * 2.0 + t * 0.9);

  color = vec4(pow(vec3(r, g, b), vec3(0.8)), 1.0);
}
