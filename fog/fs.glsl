#version 300 es
precision highp float;

in vec3 fragPosition;
in vec3 fragNormal;

uniform vec3 lightDirection;
uniform vec3 eyePosition;
uniform float fogDensity;
uniform bool fogEnabled;

out vec4 fragColor;

void main() {
  vec3 N = normalize(fragNormal);
  vec3 L = normalize(lightDirection);
  vec3 V = normalize(eyePosition - fragPosition);
  vec3 H = normalize(L + V);

  float slope = 0.7;
  bool shallow = N.y > slope;

  vec3 diffuseColor = shallow ? vec3(0.2, 0.6, 0.1) : vec3(0.6, 0.3, 0.3);
  float shine = shallow ? 64.0 : 16.0;
  float specularIntensity = shallow ? 0.8 : 0.3;

  vec3 specularColor = vec3(1.0);
  
  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(N, H), 0.0), shine);
  
  vec3 litColor = diffuseColor * diffuse + specularColor * specular * specularIntensity;

  vec3 fogColor = vec3(1.0);
  float d = 1.0 / gl_FragCoord.w;
  float v = 1.0;
  if (fogEnabled) {
    v = exp(-fogDensity * d);
    v = clamp(v, 0.0, 1.0);
  }

  vec3 color = mix(fogColor, litColor, v);
  
  fragColor = vec4(color, 1.0);
}
