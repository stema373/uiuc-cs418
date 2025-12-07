#version 300 es
precision highp float;

uniform mat4 transform;

out float isOutline;

void main() {
  const vec2 verts[18] = vec2[18](
    // top bar
    vec2(-0.4,  0.8), vec2( 0.4,  0.8), vec2(-0.4,  0.4),
    vec2(-0.4,  0.4), vec2( 0.4,  0.8), vec2( 0.4,  0.4),

    // stem
    vec2(-0.2,  0.4), vec2( 0.2,  0.4), vec2(-0.2, -0.4),
    vec2(-0.2, -0.4), vec2( 0.2,  0.4), vec2( 0.2, -0.4),

    // bottom bar
    vec2(-0.4, -0.4), vec2( 0.4, -0.4), vec2(-0.4, -0.8),
    vec2(-0.4, -0.8), vec2( 0.4, -0.4), vec2( 0.4, -0.8)
  );

  const vec2 outlineVerts[18] = vec2[18](
    // top bar
    vec2(-0.45,  0.85), vec2( 0.45,  0.85), vec2(-0.45,  0.35),
    vec2(-0.45,  0.35), vec2( 0.45,  0.85), vec2( 0.45,  0.35),

    // stem
    vec2(-0.25,  0.35), vec2( 0.25,  0.35), vec2(-0.25, -0.35),
    vec2(-0.25, -0.35), vec2( 0.25,  0.35), vec2( 0.25, -0.35),

    // bottom bar
    vec2(-0.45, -0.35), vec2( 0.45, -0.35), vec2(-0.45, -0.85),
    vec2(-0.45, -0.85), vec2( 0.45, -0.35), vec2( 0.45, -0.85)
  );

  int outline = gl_VertexID < 18 ? 1 : 0;
  int idx = gl_VertexID % 18;

  vec2 pos2 = outline == 1 ? outlineVerts[idx] : verts[idx];
  gl_Position = transform * vec4(pos2, 0.0, 1.0);

  isOutline = float(outline);
}
