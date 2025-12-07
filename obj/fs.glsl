#version 300 es
precision highp float;

in vec3 fragPosition;
in vec3 fragNormal;
in vec3 fragColorAttr;

uniform vec3 lightDirection;
uniform vec3 eyePosition;

out vec4 fragColor;

uniform vec4 RGBA;

void main() {
  vec3 diffuseColor = fragColorAttr;
  float a = RGBA.a;
  vec3 specularColor = vec3(1.0);

  vec3 N = normalize(fragNormal);
  vec3 L = normalize(lightDirection);
  vec3 V = normalize(eyePosition - fragPosition);
  vec3 H = normalize(L + V);

  float diffuse = max(dot(N, L), 0.0) * (1.0 - a);

  float specular = pow(max(dot(N, H), 0.0), 64.0);
  
  vec3 color = diffuseColor * diffuse + specularColor * specular * 3.0 * a;
  
  fragColor = vec4(color, 1.0);
}
