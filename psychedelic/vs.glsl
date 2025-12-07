#version 300 es
out vec4 pos;
void main() {
  const vec2 verts[6] = vec2[6](
      vec2(-1,-1),
      vec2( 1,-1),
      vec2(-1, 1),

      vec2(-1, 1),
      vec2( 1,-1),
      vec2( 1, 1)
  );
  gl_Position = vec4(verts[gl_VertexID],0,1);
  pos = gl_Position;
}
