#include "uselibpng.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

static constexpr double INF = 1e30;

struct Color {
  double r = 0, g = 0, b = 0, a = 1;
};

string trim(const string &s) {
  size_t start = 0;
  while (start < s.size() && isspace(static_cast<unsigned char>(s[start])))
    start++;
  size_t end = s.size();
  while (end > start && isspace(static_cast<unsigned char>(s[end - 1])))
    end--;
  return s.substr(start, end - start);
}

double linearToSRGB(double c) {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

double sRGBToLinear(double c) {
  return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

double signedArea(const vector<double> &v0, const vector<double> &v1,
                  const vector<double> &v2) {
  return ((v1[0] - v0[0]) * (v2[1] - v0[1]) -
          (v2[0] - v0[0]) * (v1[1] - v0[1])) *
         0.5;
}

vector<vector<double>> dda(vector<double> a, vector<double> b, int d) {
  vector<vector<double>> res;
  if (a[d] == b[d])
    return res;
  if (a[d] > b[d])
    swap(a, b);

  vector<double> delta(a.size());
  for (size_t i = 0; i < a.size(); i++)
    delta[i] = b[i] - a[i];

  double delta_d = delta[d];
  vector<double> step(a.size());
  for (size_t i = 0; i < a.size(); i++)
    step[i] = delta[i] / delta_d;

  double e = ceil(a[d]) - a[d];

  vector<double> offset(a.size());
  for (size_t i = 0; i < a.size(); i++)
    offset[i] = e * step[i];

  vector<double> p(a.size());
  for (size_t i = 0; i < a.size(); i++)
    p[i] = a[i] + offset[i];

  while (p[d] < b[d]) {
    res.push_back(p);
    for (size_t i = 0; i < a.size(); i++)
      p[i] += step[i];
  }

  return res;
}

void fillHorizontal(vector<double> &left, vector<double> &right,
                    vector<tuple<int, int, Color>> &pixels, bool depth,
                    vector<vector<double>> *zBuffer, bool hyp, Image *texture,
                    bool decals) {
  for (auto &v : dda(left, right, 0)) {
    int x = round(v[0]);
    int y = round(v[1]);
    double z = v[2];

    if (depth && zBuffer) {
      if (x < 0 || x >= (int)zBuffer->at(0).size() || y < 0 ||
          y >= (int)zBuffer->size())
        continue;

      if (z >= (*zBuffer)[y][x])
        continue;

      (*zBuffer)[y][x] = z;
    }

    Color c;
    if (texture) {
      double invW = v[3];
      double s = hyp ? v[8] / invW : v[8];
      double t = hyp ? v[9] / invW : v[9];

      s = s - floor(s);
      t = t - floor(t);

      int tx = round(s * texture->width());
      int ty = round(t * texture->height());

      tx = clamp(tx, 0, (int)texture->width() - 1);
      ty = clamp(ty, 0, (int)texture->height() - 1);

      if (tx >= 0 && tx < (int)texture->width() && ty >= 0 &&
          ty < (int)texture->height()) {
        auto &texel = (*texture)[ty][tx];

        double tr = sRGBToLinear(texel.r / 255.0);
        double tg = sRGBToLinear(texel.g / 255.0);
        double tb = sRGBToLinear(texel.b / 255.0);
        double ta = texel.a / 255.0;

        if (decals) {
          double vcr, vcg, vcb, vca;
          if (hyp) {
            vcr = clamp(v[4] / invW, 0.0, 1.0);
            vcg = clamp(v[5] / invW, 0.0, 1.0);
            vcb = clamp(v[6] / invW, 0.0, 1.0);
            vca = clamp(v[7] / invW, 0.0, 1.0);
          } else {
            vcr = clamp(v[4], 0.0, 1.0);
            vcg = clamp(v[5], 0.0, 1.0);
            vcb = clamp(v[6], 0.0, 1.0);
            vca = clamp(v[7], 0.0, 1.0);
          }

          double a = ta + vca - ta * vca;
          double r = tr * ta + vcr * (1.0 - ta);
          double g = tg * ta + vcg * (1.0 - ta);
          double b = tb * ta + vcb * (1.0 - ta);

          c.r = clamp(r, 0.0, 1.0);
          c.g = clamp(g, 0.0, 1.0);
          c.b = clamp(b, 0.0, 1.0);
          c.a = clamp(a, 0.0, 1.0);
        } else {
          c.r = tr;
          c.g = tg;
          c.b = tb;
          c.a = ta;
        }

        pixels.emplace_back(x, y, c);
      }
    } else {
      if (hyp) {
        double invW = v[3];
        c = Color{clamp(v[4] / invW, 0.0, 1.0), clamp(v[5] / invW, 0.0, 1.0),
                  clamp(v[6] / invW, 0.0, 1.0), clamp(v[7] / invW, 0.0, 1.0)};
      } else {
        c = Color{clamp(v[4], 0.0, 1.0), clamp(v[5], 0.0, 1.0),
                  clamp(v[6], 0.0, 1.0), clamp(v[7], 0.0, 1.0)};
      }
      pixels.emplace_back(x, y, c);
    }
  }
}

void scanline(vector<double> p, vector<double> q, vector<double> r, bool depth,
              vector<vector<double>> *zBuffer, bool hyp,
              vector<tuple<int, int, Color>> &pixels, Image *texture,
              bool decals) {
  vector<vector<double>> verts = {p, q, r};
  sort(verts.begin(), verts.end(),
       [](auto &a, auto &b) { return a[1] < b[1]; });

  auto t = verts[0], m = verts[1], b = verts[2];

  auto longEdge = dda(t, b, 1);
  if (longEdge.empty())
    return;

  auto topEdge = dda(t, m, 1);
  size_t iLong = 0, iShort = 0;
  while (iShort < topEdge.size() && iLong < longEdge.size()) {
    fillHorizontal(topEdge[iShort], longEdge[iLong], pixels, depth, zBuffer,
                   hyp, texture, decals);
    iShort++;
    iLong++;
  }

  auto bottomEdge = dda(m, b, 1);
  iShort = 0;
  while (iShort < bottomEdge.size() && iLong < longEdge.size()) {
    fillHorizontal(bottomEdge[iShort], longEdge[iLong], pixels, depth, zBuffer,
                   hyp, texture, decals);
    iShort++;
    iLong++;
  }
}

double planeDistance(const vector<double> &v, int plane) {
  double x = v[0], y = v[1], z = v[2], w = v[3];
  switch (plane) {
  case 0:
    return w + x;
  case 1:
    return w - x;
  case 2:
    return w + y;
  case 3:
    return w - y;
  case 4:
    return w + z;
  case 5:
    return w - z;
  }
  return 0;
}

vector<double> intersectEdge(const vector<double> &a, const vector<double> &b,
                             int plane) {
  double da = planeDistance(a, plane);
  double db = planeDistance(b, plane);

  double t = da / (da - db);

  vector<double> out(a.size());
  for (size_t i = 0; i < a.size(); i++)
    out[i] = a[i] + t * (b[i] - a[i]);

  return out;
}

vector<vector<vector<double>>>
clipTriangleAgainstPlane(const vector<vector<double>> &tri, int plane,
                         bool hyp) {
  vector<vector<vector<double>>> res;
  const vector<double> &v0 = tri[0], &v1 = tri[1], &v2 = tri[2];

  double d0 = planeDistance(v0, plane);
  double d1 = planeDistance(v1, plane);
  double d2 = planeDistance(v2, plane);

  bool in0 = d0 >= 0, in1 = d1 >= 0, in2 = d2 >= 0;
  int cnt = in0 + in1 + in2;

  if (cnt == 0) {
    return res;
  } else if (cnt == 3) {
    res.push_back(tri);
    return res;
  } else if (cnt == 1) {
    vector<double> in, out0, out1;
    if (in0) {
      in = v0;
      out0 = v1;
      out1 = v2;
    } else if (in1) {
      in = v1;
      out0 = v2;
      out1 = v0;
    } else {
      in = v2;
      out0 = v0;
      out1 = v1;
    }
    auto i0 = intersectEdge(in, out0, plane);
    auto i1 = intersectEdge(in, out1, plane);

    res.push_back({in, i0, i1});
    return res;
  } else if (cnt == 2) {
    vector<double> in0v, in1v, out;
    if (!in0) {
      out = v0;
      in0v = v1;
      in1v = v2;
    } else if (!in1) {
      out = v1;
      in0v = v2;
      in1v = v0;
    } else {
      out = v2;
      in0v = v0;
      in1v = v1;
    }
    auto i0 = intersectEdge(in0v, out, plane);
    auto i1 = intersectEdge(in1v, out, plane);

    res.push_back({in0v, in1v, i1});
    res.push_back({in0v, i1, i0});

    return res;
  }
  return res;
}

vector<vector<vector<double>>>
clipTriangleFrustum(const vector<vector<double>> &tri, bool hyp) {
  vector<vector<vector<double>>> triangles = {tri};

  for (int plane = 0; plane < 6; plane++) {
    vector<vector<vector<double>>> next;
    for (const auto &t : triangles) {
      auto clipped = clipTriangleAgainstPlane(t, plane, hyp);
      next.insert(next.end(), clipped.begin(), clipped.end());
    }
    triangles = std::move(next);

    if (triangles.empty())
      break;
  }

  return triangles;
}

vector<double> transformVertex(const vector<double> &pos,
                               const double *matrix) {
  vector<double> v(4, 0.0);
  for (int i = 0; i < 4; i++)
    v[i] += matrix[i] * pos[0] + matrix[i + 4] * pos[1] +
            matrix[i + 8] * pos[2] + matrix[i + 12] * pos[3];
  return v;
}

vector<double> project(const vector<double> &v, int width, int height,
                       bool hyp) {
  if (hyp) {
    return {(v[0] / v[3] + 1) * 0.5 * width,
            (v[1] / v[3] + 1) * 0.5 * height,
            v[2] / v[3],
            1.0 / v[3],
            v[4] / v[3],
            v[5] / v[3],
            v[6] / v[3],
            v[7] / v[3],
            v[8] / v[3],
            v[9] / v[3]};
  } else {
    return {(v[0] / v[3] + 1) * 0.5 * width,
            (v[1] / v[3] + 1) * 0.5 * height,
            v[2] / v[3],
            1.0 / v[3],
            v[4],
            v[5],
            v[6],
            v[7],
            v[8],
            v[9]};
  }
}

vector<vector<double>> buildVertices(const vector<vector<double>> &positions,
                                     const vector<vector<double>> &colors,
                                     const vector<vector<double>> &texcoords,
                                     const double *matrix) {
  vector<vector<double>> vertex;
  size_t n = positions.size();
  for (size_t i = 0; i < n; i++) {
    vector<double> v(10, 0.0);
    auto t = transformVertex(positions[i], matrix);
    double invW = 1.0 / t[3];
    copy(t.begin(), t.end(), v.begin());

    if (!colors.empty()) {
      v[4] = colors[i][0];
      v[5] = colors[i][1];
      v[6] = colors[i][2];
      v[7] = colors[i][3];
    }

    if (i < texcoords.size()) {
      v[8] = texcoords[i][0];
      v[9] = texcoords[i][1];
    }
    vertex.push_back(v);
  }
  return vertex;
}

void drawArraysTriangles(int width, int height, Image *texture,
                         const vector<vector<double>> &positions,
                         const vector<vector<double>> &colors,
                         const vector<vector<double>> &texcoords,
                         const double *matrix, int first, int count, bool depth,
                         vector<vector<double>> *zBuffer, bool hyp, bool cull,
                         bool decals, bool frustum,
                         vector<tuple<int, int, Color>> &pixels) {
  vector<int> indices(count);
  for (int i = 0; i < count; i++)
    indices[i] = first + i;

  auto vertex = buildVertices(positions, colors, texcoords, matrix);

  if (depth && zBuffer && zBuffer->empty())
    zBuffer->assign(height, vector<double>(width, INF));

  for (size_t i = 0; i + 2 < indices.size(); i += 3) {
    vector<vector<vector<double>>> tris;
    auto &v0 = vertex[indices[i]];
    auto &v1 = vertex[indices[i + 1]];
    auto &v2 = vertex[indices[i + 2]];

    if (frustum) {
      tris = clipTriangleFrustum({v0, v1, v2}, hyp);
    } else {
      tris.push_back({v0, v1, v2});
    }

    for (auto &tri : tris) {
      auto v0 = project(tri[0], width, height, hyp);
      auto v1 = project(tri[1], width, height, hyp);
      auto v2 = project(tri[2], width, height, hyp);

      if (cull && signedArea(v0, v1, v2) >= 0)
        continue;

      scanline(v0, v1, v2, depth, zBuffer, hyp, pixels, texture, decals);
    }
  }
}

void drawElementsTriangles(int width, int height, Image *texture,
                           const vector<vector<double>> &positions,
                           const vector<vector<double>> &colors,
                           const vector<vector<double>> &texcoords,
                           const double *matrix, const vector<int> &elements,
                           int count, int offset, bool depth,
                           vector<vector<double>> *zBuffer, bool hyp, bool cull,
                           bool decals, bool frustum,
                           vector<tuple<int, int, Color>> &pixels) {
  vector<int> indices(count);
  for (int i = 0; i < count; i++)
    indices[i] = elements[offset + i];

  auto vertex = buildVertices(positions, colors, texcoords, matrix);

  if (depth && zBuffer && zBuffer->empty())
    zBuffer->assign(height, vector<double>(width, INF));

  for (size_t i = 0; i + 2 < indices.size(); i += 3) {
    vector<vector<vector<double>>> tris;
    auto &v0 = vertex[indices[i]];
    auto &v1 = vertex[indices[i + 1]];
    auto &v2 = vertex[indices[i + 2]];

    if (frustum) {
      tris = clipTriangleFrustum({v0, v1, v2}, hyp);
    } else {
      tris.push_back({v0, v1, v2});
    }

    for (auto &tri : tris) {
      auto v0 = project(tri[0], width, height, hyp);
      auto v1 = project(tri[1], width, height, hyp);
      auto v2 = project(tri[2], width, height, hyp);

      if (cull && signedArea(v0, v1, v2) >= 0)
        continue;

      scanline(v0, v1, v2, depth, zBuffer, hyp, pixels, texture, decals);
    }
  }
}

void drawArraysPoints(int width, int height, Image *texture,
                      const vector<vector<double>> &positions,
                      const vector<vector<double>> &colors,
                      const vector<vector<double>> &texcoords,
                      const double *matrix, const vector<double> &pointsizes,
                      int first, int count, bool depth,
                      vector<vector<double>> *zBuffer, bool hyp, bool cull,
                      bool decals, bool frustum,
                      vector<tuple<int, int, Color>> &pixels) {
  auto vertex = buildVertices(positions, colors, texcoords, matrix);

  if (depth && zBuffer && zBuffer->empty())
    zBuffer->assign(height, vector<double>(width, INF));

  for (int i = 0; i < count; i++) {
    vector<vector<double>> tris;
    int idx = first + i;

    auto &v = vertex[idx];
    auto p = project(v, width, height, hyp);

    double half = pointsizes[idx] * 0.5;

    vector<double> p0, p1, p2, p3;
    p0 = p1 = p2 = p3 = p;

    p0[0] -= half;
    p0[1] -= half;
    p0[8] = 0;
    p0[9] = 0;

    p1[0] += half;
    p1[1] -= half;
    p1[8] = 1;
    p1[9] = 0;

    p2[0] += half;
    p2[1] += half;
    p2[8] = 1;
    p2[9] = 1;

    p3[0] -= half;
    p3[1] += half;
    p3[8] = 0;
    p3[9] = 1;

    scanline(p0, p1, p2, depth, zBuffer, hyp, pixels, texture, decals);
    scanline(p0, p2, p3, depth, zBuffer, hyp, pixels, texture, decals);
  }
}

enum class DrawType { ARRAYS, ELEMENTS, POINTS };

struct DrawCommand {
  DrawType type;

  vector<vector<double>> positions;
  vector<vector<double>> colors;
  vector<vector<double>> texcoords;
  vector<double> pointsizes;
  vector<int> elements;

  double matrix[16];

  int first;
  int count;
  int offset;

  shared_ptr<Image> texture;

  bool depth;
  bool hyp;
  bool cull;
  bool decals;
  bool frustum;
};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <filename>\n";
    return 1;
  }

  string filename = argv[1];

  ifstream file(filename);
  if (!file) {
    cerr << "Could not open file: " << filename << "\n";
    return 1;
  }

  uint32_t width = 0, height = 0;
  string outFilename;
  bool depth = false, sRGB = false, hyp = false, cull = false, decals = false,
       frustum = false;
  vector<vector<double>> zBuffer;
  int fsaa = 1;
  shared_ptr<Image> texture;
  double matrix[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  vector<vector<double>> positions, colors, texcoords, vertex;
  vector<double> pointsizes;
  vector<int> elements;
  vector<tuple<int, int, Color>> pixels;

  vector<DrawCommand> commands;

  string line;
  while (getline(file, line)) {
    line = trim(line);
    if (line.empty())
      continue;

    istringstream iss(line);
    string keyword;
    iss >> keyword;

    if (keyword == "png") {
      iss >> width >> height >> outFilename;
    } else if (keyword == "depth") {
      depth = true;
    } else if (keyword == "sRGB") {
      sRGB = true;
    } else if (keyword == "hyp") {
      hyp = true;
    } else if (keyword == "fsaa") {
      iss >> fsaa;
    } else if (keyword == "cull") {
      cull = true;
    } else if (keyword == "decals") {
      decals = true;
    } else if (keyword == "frustum") {
      frustum = true;
    } else if (keyword == "texture") {
      string texFilename;
      if (iss >> texFilename) {
        texture = make_shared<Image>(texFilename.c_str());
      }
    } else if (keyword == "uniformMatrix") {
      for (int i = 0; i < 16; i++) {
        iss >> matrix[i];
      }
    } else if (keyword == "position") {
      int p_size;
      iss >> p_size;
      positions.clear();

      while (true) {
        vector<double> p = {0, 0, 0, 1};
        bool ok = true;
        for (int i = 0; i < p_size; i++) {
          if (!(iss >> p[i])) {
            ok = false;
            break;
          }
        }

        if (!ok)
          break;

        positions.push_back(p);
      }
    } else if (keyword == "color") {
      int c_size;
      iss >> c_size;
      colors.clear();

      while (true) {
        vector<double> c = {0, 0, 0, 1};
        bool ok = true;

        for (int i = 0; i < c_size; i++) {
          if (!(iss >> c[i])) {
            ok = false;
            break;
          }
        }

        if (!ok)
          break;

        colors.push_back(c);
      }
    } else if (keyword == "texcoord") {
      int t_size;
      iss >> t_size;
      texcoords.clear();

      while (true) {
        vector<double> t(2);
        bool ok = true;

        for (int i = 0; i < t_size; i++) {
          if (!(iss >> t[i])) {
            ok = false;
            break;
          }
        }

        if (!ok)
          break;

        texcoords.push_back(t);
      }
    } else if (keyword == "pointsize") {
      int p_size;
      iss >> p_size;
      pointsizes.clear();

      double i;
      while (iss >> i)
        pointsizes.push_back(i);
    } else if (keyword == "elements") {
      elements.clear();
      int i;
      while (iss >> i)
        elements.push_back(i);
    } else if (keyword == "drawArraysTriangles") {
      int first, count;
      iss >> first >> count;

      DrawCommand cmd;
      cmd.type = DrawType::ARRAYS;
      cmd.positions = positions;
      cmd.colors = colors;
      cmd.texcoords = texcoords;
      cmd.pointsizes = pointsizes;
      cmd.elements = elements;
      copy(begin(matrix), end(matrix), cmd.matrix);
      cmd.first = first;
      cmd.count = count;
      cmd.offset = 0;
      cmd.texture = texture;
      cmd.depth = depth;
      cmd.hyp = hyp;
      cmd.cull = cull;
      cmd.decals = decals;
      cmd.frustum = frustum;

      commands.push_back(std::move(cmd));
    } else if (keyword == "drawElementsTriangles") {
      int count, offset;
      iss >> count >> offset;

      DrawCommand cmd;
      cmd.type = DrawType::ELEMENTS;
      cmd.positions = positions;
      cmd.colors = colors;
      cmd.texcoords = texcoords;
      cmd.pointsizes = pointsizes;
      cmd.elements = elements;
      copy(begin(matrix), end(matrix), cmd.matrix);
      cmd.first = 0;
      cmd.count = count;
      cmd.offset = offset;
      cmd.texture = texture;
      cmd.depth = depth;
      cmd.hyp = hyp;
      cmd.cull = cull;
      cmd.decals = decals;
      cmd.frustum = frustum;

      commands.push_back(std::move(cmd));
    } else if (keyword == "drawArraysPoints") {
      int first, count;
      iss >> first >> count;

      DrawCommand cmd;
      cmd.type = DrawType::POINTS;
      cmd.positions = positions;
      cmd.colors = colors;
      cmd.texcoords = texcoords;
      cmd.pointsizes = pointsizes;
      cmd.elements = elements;
      copy(begin(matrix), end(matrix), cmd.matrix);
      cmd.first = first;
      cmd.count = count;
      cmd.offset = 0;
      cmd.texture = texture;
      cmd.depth = depth;
      cmd.hyp = hyp;
      cmd.cull = cull;
      cmd.decals = decals;
      cmd.frustum = frustum;

      commands.push_back(std::move(cmd));
    }
  }

  int w = width * fsaa;
  int h = height * fsaa;

  if (depth)
    zBuffer.assign(h, vector<double>(w, INF));

  for (const auto &cmd : commands) {
    if (cmd.type == DrawType::ARRAYS) {
      drawArraysTriangles(w, h, cmd.texture.get(), cmd.positions, cmd.colors,
                          cmd.texcoords, cmd.matrix, cmd.first, cmd.count,
                          cmd.depth, &zBuffer, cmd.hyp, cmd.cull, cmd.decals,
                          cmd.frustum, pixels);
    } else if (cmd.type == DrawType::ELEMENTS) {
      drawElementsTriangles(w, h, cmd.texture.get(), cmd.positions, cmd.colors,
                            cmd.texcoords, cmd.matrix, cmd.elements, cmd.count,
                            cmd.offset, cmd.depth, &zBuffer, cmd.hyp, cmd.cull,
                            cmd.decals, cmd.frustum, pixels);
    } else if (cmd.type == DrawType::POINTS) {
      drawArraysPoints(w, h, cmd.texture.get(), cmd.positions, cmd.colors,
                       cmd.texcoords, cmd.matrix, cmd.pointsizes, cmd.first,
                       cmd.count, cmd.depth, &zBuffer, cmd.hyp, cmd.cull,
                       cmd.decals, cmd.frustum, pixels);
    }
  }

  vector<vector<Color>> buffer(h, vector<Color>(w, {0, 0, 0, 0}));
  vector<vector<vector<double>>> accum(
      h, vector<vector<double>>(w, vector<double>{0, 0, 0, 0}));

  for (auto &frag : pixels) {
    int x, y;
    Color c;
    tie(x, y, c) = frag;
    if (x < 0 || x >= w || y < 0 || y >= h)
      continue;

    double rs = clamp(c.r, 0.0, 1.0), gs = clamp(c.g, 0.0, 1.0),
           bs = clamp(c.b, 0.0, 1.0), as = clamp(c.a, 0.0, 1.0);

    auto &dst = accum[y][x];
    double rd = dst[0], gd = dst[1], bd = dst[2], ad = dst[3];

    double a = as + ad * (1.0 - as);
    double r = 0.0, g = 0.0, b = 0.0;
    if (a > 0.0) {
      r = (as * rs + (1.0 - as) * ad * rd) / a;
      g = (as * gs + (1.0 - as) * ad * gd) / a;
      b = (as * bs + (1.0 - as) * ad * bd) / a;
    }

    dst[0] = r;
    dst[1] = g;
    dst[2] = b;
    dst[3] = a;
  }

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      auto v = accum[y][x];
      double r = clamp(v[0], 0.0, 1.0);
      double g = clamp(v[1], 0.0, 1.0);
      double b = clamp(v[2], 0.0, 1.0);
      double a = clamp(v[3], 0.0, 1.0);

      buffer[y][x] = {r, g, b, a};
    }
  }

  unique_ptr<Image> img = make_unique<Image>((uint32_t)width, (uint32_t)height);

  for (uint32_t y = 0; y < height; y++) {
    for (uint32_t x = 0; x < width; x++) {
      double sr = 0.0, sg = 0.0, sb = 0.0, sa = 0.0;
      for (int yy = 0; yy < fsaa; yy++) {
        for (int xx = 0; xx < fsaa; xx++) {
          int rx = (int)x * fsaa + xx;
          int ry = (int)y * fsaa + yy;
          double r = buffer[ry][rx].r;
          double g = buffer[ry][rx].g;
          double b = buffer[ry][rx].b;
          double a = buffer[ry][rx].a;

          sr += r * a;
          sg += g * a;
          sb += b * a;
          sa += a;
        }
      }

      double samples = (double)(fsaa * fsaa);
      double a = sa / samples;
      double r = sr / samples / a;
      double g = sg / samples / a;
      double b = sb / samples / a;

      if (sRGB) {
        r = linearToSRGB(r);
        g = linearToSRGB(g);
        b = linearToSRGB(b);
      }

      (*img)[y][x].r = (uint8_t)(clamp(r, 0.0, 1.0) * 255);
      (*img)[y][x].g = (uint8_t)(clamp(g, 0.0, 1.0) * 255);
      (*img)[y][x].b = (uint8_t)(clamp(b, 0.0, 1.0) * 255);
      (*img)[y][x].a = (uint8_t)(clamp(a, 0.0, 1.0) * 255);
    }
  }

  img->save(outFilename.c_str());

  return 0;
}
