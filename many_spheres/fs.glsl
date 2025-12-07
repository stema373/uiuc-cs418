#version 300 es
precision highp float;

in vec3 sphereColor;
in vec3 viewSpaceCenter;
in float sphereRadius;

uniform vec3 lightDirection;
uniform mat4 mv;

out vec4 fragColor;

void main() {
  vec2 xy = gl_PointCoord * 2.0 - 1.0;

  float xyLength = length(xy);
  if (xyLength > 1.0) discard;

  float nz = sqrt(1.0 - xyLength * xyLength);
  vec3 N = vec3(xy, nz);

  vec3 surfacePos = viewSpaceCenter + N * sphereRadius;

  vec3 L = normalize((mv * vec4(lightDirection, 0.0)).xyz);

  vec3 V = normalize(-surfacePos);
  vec3 H = normalize(L + V);

  float diffuse = max(dot(N, L), 0.0);
  float specular = pow(max(dot(N, H), 0.0), 64.0);
  vec3 specularColor = vec3(1.0);

  vec3 color = sphereColor * diffuse + specularColor * specular;

  fragColor = vec4(color, 1.0);
}
