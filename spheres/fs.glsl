#version 300 es
precision highp float;

in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 lightDirection;
uniform vec3 eyePosition;
uniform vec3 diffuseColor;

out vec4 fragColor;

void main() {
  vec3 specularColor = vec3(1.0);

  vec3 N = normalize(fragNormal);
  vec3 L = normalize(lightDirection);
  vec3 V = normalize(eyePosition - fragPosition);
  vec3 H = normalize(L + V);

  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(N, H), 0.0), 64.0);
  
  vec3 color = diffuseColor * diffuse + specularColor * specular;
  
  fragColor = vec4(color, 1.0);
}
