#version 300 es
precision highp float;

in vec3 fragNormal;
in vec2 fragTexCoord;
in vec3 fragColorAttr;

uniform vec3 lightDirection;
uniform sampler2D u_texture;

out vec4 fragColor;

void main() {
  vec3 diffuseColor = texture(u_texture, fragTexCoord).rgb;

  vec3 N = normalize(fragNormal);
  vec3 L = normalize(lightDirection);

  float diffuse = max(dot(N, L), 0.0);

  vec3 color = diffuseColor * diffuse;
  fragColor = vec4(color, 1.0);
}
