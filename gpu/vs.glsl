#version 300 es
precision highp float;

uniform float seconds;

out float outline;

void main() {
  float s = seconds * 42.0;
  vec2 iPoints[12] = vec2[12](
    vec2(-0.4 + 0.1 * cos(0.84 * s),  0.8 + 0.1 * sin(0.84 * s)), vec2( 0.4 + 0.1 * cos(0.17 * s),  0.8 + 0.1 * sin(0.17 * s)), vec2( 0.4 + 0.1 * cos(0.88 * s),  0.4 + 0.1 * sin(0.88 * s)), vec2(-0.4 + 0.1 * cos(0.08 * s),  0.4 + 0.1 * sin(0.08 * s)),

    vec2(-0.2 + 0.1 * cos(0.40 * s),  0.4 + 0.1 * sin(0.40 * s)), vec2( 0.2 + 0.1 * cos(0.20 * s),  0.4 + 0.1 * sin(0.20 * s)), vec2( 0.2 + 0.1 * cos(0.12 * s), -0.4 + 0.1 * sin(0.12 * s)), vec2(-0.2 + 0.1 * cos(0.98 * s), -0.4 + 0.1 * sin(0.98 * s)),

    vec2(-0.4 + 0.1 * cos(0.19 * s), -0.4 + 0.1 * sin(0.19 * s)), vec2( 0.4 + 0.1 * cos(0.75 * s), -0.4 + 0.1 * sin(0.85 * s)), vec2( 0.4 + 0.1 * cos(0.68 * s), -0.8 + 0.1 * sin(0.68 * s)), vec2(-0.4 + 0.1 * cos(0.60 * s), -0.8 + 0.1 * sin(0.60 * s))
  );

  vec2 oPoints[24] = vec2[24](
    vec2(-0.45 + 0.1 * cos(0.42 * s),  0.85 + 0.1 * sin(0.42 * s)), vec2( 0.45 + 0.1 * cos(0.04 * s),  0.85 + 0.1 * sin(0.04 * s)),

    vec2(-0.40 + 0.1 * cos(0.01 * s),  0.80 + 0.1 * sin(0.01 * s)), vec2( 0.40 + 0.1 * cos(0.64 * s),  0.80 + 0.1 * sin(0.64 * s)),

    vec2(-0.40 + 0.1 * cos(0.81 * s),  0.40 + 0.1 * sin(0.81 * s)), vec2(-0.20 + 0.1 * cos(0.44 * s),  0.40 + 0.1 * sin(0.44 * s)), vec2( 0.20 + 0.1 * cos(0.16 * s),  0.40 + 0.1 * sin(0.16 * s)), vec2( 0.40 + 0.1 * cos(0.09 * s),  0.40 + 0.1 * sin(0.09 * s)),

    vec2(-0.45 + 0.1 * cos(0.05 * s),  0.35 + 0.1 * sin(0.05 * s)), vec2(-0.25 + 0.1 * cos(0.65 * s),  0.35 + 0.1 * sin(0.65 * s)), vec2( 0.25 + 0.1 * cos(0.02 * s),  0.35 + 0.1 * sin(0.02 * s)), vec2( 0.45 + 0.1 * cos(0.18 * s),  0.35 + 0.1 * sin(0.18 * s)),

    vec2(-0.45 + 0.1 * cos(0.92 * s), -0.35 + 0.1 * sin(0.92 * s)), vec2(-0.25 + 0.1 * cos(0.60 * s), -0.35 + 0.1 * sin(0.60 * s)), vec2( 0.25 + 0.1 * cos(0.93 * s), -0.35 + 0.1 * sin(0.93 * s)), vec2( 0.45 + 0.1 * cos(0.30 * s), -0.35 + 0.1 * sin(0.30 * s)),

    vec2(-0.40 + 0.1 * cos(0.59 * s), -0.40 + 0.1 * sin(0.59 * s)), vec2(-0.20 + 0.1 * cos(0.88 * s), -0.40 + 0.1 * sin(0.88 * s)), vec2( 0.20 + 0.1 * cos(0.23 * s), -0.40 + 0.1 * sin(0.23 * s)), vec2( 0.40 + 0.1 * cos(0.35 * s), -0.4 + 0.1 * sin(0.35 * s)),

    vec2(-0.40 + 0.1 * cos(0.05 * s), -0.80 + 0.1 * sin(0.05 * s)), vec2( 0.40 + 0.1 * cos(0.76 * s), -0.80 + 0.1 * sin(0.76 * s)),

    vec2(-0.45 + 0.1 * cos(0.90 * s), -0.85 + 0.1 * sin(0.90 * s)), vec2( 0.45 + 0.1 * cos(0.62 * s), -0.85 + 0.1 * sin(0.62 * s))
  );

  vec2 verts[102] = vec2[102](
    // top bar
    iPoints[0], iPoints[4], iPoints[3],
    iPoints[0], iPoints[1], iPoints[4],
    iPoints[4], iPoints[1], iPoints[5],
    iPoints[5], iPoints[1], iPoints[2],

    // stem
    iPoints[4], iPoints[5], iPoints[7],
    iPoints[7], iPoints[5], iPoints[6],

    // bottom bar
    iPoints[8], iPoints[7], iPoints[11],
    iPoints[7], iPoints[6], iPoints[11],
    iPoints[10], iPoints[11], iPoints[6],
    iPoints[6], iPoints[9], iPoints[10],

    // outline
    oPoints[0], oPoints[1], oPoints[2],
    oPoints[2], oPoints[1], oPoints[3],
    oPoints[1], oPoints[11], oPoints[3],
    oPoints[3], oPoints[11], oPoints[7],
    oPoints[11], oPoints[10], oPoints[7],
    oPoints[7], oPoints[10], oPoints[6],
    oPoints[10], oPoints[14], oPoints[6],
    oPoints[6], oPoints[14], oPoints[18],
    oPoints[14], oPoints[15], oPoints[18],
    oPoints[18], oPoints[15], oPoints[19],
    oPoints[15], oPoints[23], oPoints[19],
    oPoints[19], oPoints[23], oPoints[21],
    oPoints[23], oPoints[22], oPoints[21],
    oPoints[21], oPoints[22], oPoints[20],
    oPoints[22], oPoints[12], oPoints[20],
    oPoints[20], oPoints[12], oPoints[16],
    oPoints[12], oPoints[13], oPoints[16],
    oPoints[16], oPoints[13], oPoints[17],
    oPoints[13], oPoints[9], oPoints[17],
    oPoints[17], oPoints[9], oPoints[5],
    oPoints[9], oPoints[8], oPoints[5],
    oPoints[5], oPoints[8], oPoints[4],
    oPoints[8], oPoints[0], oPoints[4],
    oPoints[4], oPoints[0], oPoints[2]
  );

  outline = gl_VertexID > 29 ? 1.0 : 0.0;

  gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
}
