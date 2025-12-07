#version 300 es
precision highp float;

in vec3 fragPosition;
in vec3 fragNormal;
in float fragHeight;

uniform vec3 lightDirection;
uniform vec3 eyePosition;

out vec4 fragColor;

vec3 rainbow(float height) {
  float t = fract((height + 1.0) * 0.5 * 4.5);

  vec3 colors[7] = vec3[7](
    vec3(1.0, 0.0, 0.0),    // red
    vec3(1.0, 0.5, 0.0),    // orange
    vec3(1.0, 1.0, 0.0),    // yellow
    vec3(0.0, 1.0, 0.0),    // green
    vec3(0.0, 0.0, 1.0),    // blue
    vec3(0.294, 0.0, 0.51), // indigo
    vec3(0.58, 0.0, 0.827)  // violet
  );

  float scaled = t * 6.0;
  int i = int(floor(scaled));
  float localT = fract(scaled);

  return mix(colors[i], colors[(i + 1) % 7], localT);
}

void main() {
  vec3 diffuseColor = rainbow(fragHeight);
  vec3 specularColor = vec3(1.0);
  
  vec3 N = normalize(fragNormal);
  vec3 L = normalize(lightDirection);
  vec3 V = normalize(eyePosition - fragPosition);
  vec3 H = normalize(L + V);
  
  float diffuse = max(dot(N, L), 0.0);
  
  float specular = pow(max(dot(N, H), 0.0), 32.0);
  
  vec3 color = diffuseColor * diffuse + specularColor * specular * 0.5;
  
  fragColor = vec4(color, 1.0);
}
