#include "uselibpng.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr double EPS = 1e-6;
static constexpr double INF = 1e30;

struct Vec3 {
  double x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(double a, double b, double c) : x(a), y(b), z(c) {}
  double &operator[](size_t i) { return (i == 0 ? x : (i == 1 ? y : z)); }
  double operator[](size_t i) const { return (i == 0 ? x : (i == 1 ? y : z)); }
};

struct Vec2 {
  double u = 0, v = 0;
  Vec2() = default;
  Vec2(double a, double b) : u(a), v(b) {}
};

inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 operator*(const Vec3 &a, double s) {
  return {a.x * s, a.y * s, a.z * s};
}
inline Vec3 operator*(double s, const Vec3 &a) { return a * s; }
inline Vec3 mulc(const Vec3 &a, const Vec3 &b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
inline double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double length(const Vec3 &v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3 &v) {
  double l = length(v);
  return l > EPS ? v * (1.0 / l) : Vec3{0, 0, 0};
}
inline double clamp01(double x) { return std::clamp(x, 0.0, 1.0); }
inline double linearToSRGB(double c) {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
}
inline double sRGBToLinear(double c) {
  return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

static inline std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])))
    ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
    --b;
  return s.substr(a, b - a);
}

struct Ray {
  Vec3 o, d;
  Vec3 invd;
  int sign[3];
  Ray() = default;
  Ray(const Vec3 &o_, const Vec3 &d_) : o(o_), d(d_) {
    invd.x = (std::abs(d.x) < 1e-12) ? 1e12 : 1.0 / d.x;
    invd.y = (std::abs(d.y) < 1e-12) ? 1e12 : 1.0 / d.y;
    invd.z = (std::abs(d.z) < 1e-12) ? 1e12 : 1.0 / d.z;
    sign[0] = (invd.x < 0);
    sign[1] = (invd.y < 0);
    sign[2] = (invd.z < 0);
  }
};

struct TexData {
  int w = 0, h = 0;
  std::vector<Vec3> linear;
  Vec3 sample(int x, int y) const {
    x = std::clamp(x, 0, w - 1);
    y = std::clamp(y, 0, h - 1);
    return linear[y * w + x];
  }
};

struct Light {
  Vec3 dir;
  Vec3 color;
};

struct Sphere {
  Vec3 pos, color = {1, 1, 1};
  double radius = 0, rough = 0, ior = 1.458;
  Vec3 shininess = {0, 0, 0}, transparency = {0, 0, 0};
  std::shared_ptr<TexData> tex;
};

struct Plane {
  Vec3 n, color = {1, 1, 1};
  double d = 0, rough = 0, ior = 1.458;
  Vec3 shininess = {0, 0, 0}, transparency = {0, 0, 0};
};

struct Tri {
  int i0 = 0, i1 = 0, i2 = 0;
  Vec3 p0, p1, p2, color = {1, 1, 1}, normal, a1, a2, e1, e2;
  double rough = 0, ior = 1.458;
  Vec3 shininess = {0, 0, 0}, transparency = {0, 0, 0};
  std::shared_ptr<TexData> tex;

  void compute_pre() {
    normal = normalize(cross(p1 - p0, p2 - p0));
    a1 = cross(p2 - p0, normal);
    a2 = cross(p1 - p0, normal);
    double d1 = dot(a1, (p1 - p0)), d2 = dot(a2, (p2 - p0));
    e1 = std::abs(d1) > EPS ? a1 * (1.0 / d1) : Vec3{0, 0, 0};
    e2 = std::abs(d2) > EPS ? a2 * (1.0 / d2) : Vec3{0, 0, 0};
  }
};

struct Hit {
  double t = INF;
  Vec3 position;
  double b0 = 0, b1 = 0, b2 = 0;
  int index = -1;
};

struct RayTraceResult {
  Vec3 color;
  bool hit;
};

enum class HitKind { None = -1, Sphere = 0, Plane = 1, Tri = 2 };

struct SceneHit {
  HitKind kind = HitKind::None;
  Hit hit;
  int index = -1;
};

inline bool intersect_sphere(const Ray &r, const Sphere &s, Hit &out) {
  Vec3 ro = r.o, rd = r.d, c = s.pos;
  bool inside = dot(c - ro, c - ro) < s.radius * s.radius;

  double tc = dot(c - ro, rd);
  if (!inside && tc < 0)
    return false;

  Vec3 q = ro + rd * tc;
  double d2 = dot(q - c, q - c);
  if (!inside && d2 > s.radius * s.radius)
    return false;

  double t = inside ? (tc + std::sqrt(std::max(0.0, s.radius * s.radius - d2)))
                    : (tc - std::sqrt(std::max(0.0, s.radius * s.radius - d2)));

  out.t = t;
  out.position = ro + rd * t;
  return true;
}

inline bool intersect_plane(const Ray &r, const Plane &p, Hit &out) {
  double denom = dot(r.d, p.n);
  if (std::abs(denom) < EPS)
    return false;

  double t = -(dot(r.o, p.n) + p.d) / denom;
  if (t <= 0)
    return false;

  out.t = t;
  out.position = r.o + r.d * t;
  return true;
}

inline bool intersect_triangle(const Ray &r, const Tri &t, Hit &out) {
  double denom = dot(r.d, t.normal);
  if (std::abs(denom) < EPS)
    return false;

  double tval = dot(t.p0 - r.o, t.normal) / denom;
  if (tval <= 0)
    return false;

  Vec3 p = r.o + r.d * tval, rel = p - t.p0;
  double bb1 = dot(t.e1, rel), bb2 = dot(t.e2, rel), bb0 = 1.0 - bb1 - bb2;

  if (bb0 < 0 || bb1 < 0 || bb2 < 0)
    return false;

  out.t = tval;
  out.position = p;
  out.b0 = bb0;
  out.b1 = bb1;
  out.b2 = bb2;
  return true;
}

inline Vec3 sample_texture(const std::shared_ptr<TexData> &tex, double u,
                           double v) {
  return tex ? tex->sample(int(u * tex->w + 0.5), int(v * tex->h + 0.5))
             : Vec3{0, 0, 0};
}

struct Xoroshiro128Plus {
  uint64_t s[2];

  Xoroshiro128Plus() {
    std::random_device rd;
    s[0] = rd();
    s[1] = rd();
    if (s[0] == 0 && s[1] == 0)
      s[0] = 1;
  }

  inline uint64_t next() {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s[1] = rotl(s1, 37);

    return result;
  }

  static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  inline double next_double() { return (next() >> 11) * 0x1.0p-53; }
};

static thread_local Xoroshiro128Plus rng;
inline double randf() { return rng.next_double(); }

inline Vec3 perturb_normal(const Vec3 &n, double sigma) {
  if (sigma == 0)
    return n;
  // Box-Muller transform for normal distribution
  double u1 = randf();
  double u2 = randf();
  if (u1 < 1e-9)
    u1 = 1e-9;
  double r = std::sqrt(-2.0 * std::log(u1));
  double theta = 2.0 * M_PI * u2;
  double x = r * std::cos(theta);
  double y = r * std::sin(theta);
  double z = r * std::cos(theta + M_PI / 2.0);
  double u3 = randf();
  double u4 = randf();
  if (u3 < 1e-9)
    u3 = 1e-9;
  double r2 = std::sqrt(-2.0 * std::log(u3));
  double z_val = r2 * std::cos(2.0 * M_PI * u4);

  return normalize(n + Vec3{x, y, z_val} * sigma);
}

struct AABB {
  Vec3 min, max;

  AABB() : min(INF, INF, INF), max(-INF, -INF, -INF) {}
  AABB(const Vec3 &mi, const Vec3 &ma) : min(mi), max(ma) {}

  void expand(const Vec3 &p) {
    for (int i = 0; i < 3; i++) {
      if (p[i] < min[i])
        min[i] = p[i];
      if (p[i] > max[i])
        max[i] = p[i];
    }
  }

  void expand(const AABB &box) {
    expand(box.min);
    expand(box.max);
  }

  const Vec3 &operator[](int i) const { return (i == 0) ? min : max; }
};

inline bool intersectAABB(const AABB &b, const Ray &r, double tMin,
                          double tMax) {
  double tmin, tmax, tymin, tymax, tzmin, tzmax;

  tmin = (b[r.sign[0]].x - r.o.x) * r.invd.x;
  tmax = (b[1 - r.sign[0]].x - r.o.x) * r.invd.x;
  tymin = (b[r.sign[1]].y - r.o.y) * r.invd.y;
  tymax = (b[1 - r.sign[1]].y - r.o.y) * r.invd.y;

  if ((tmin > tymax) || (tymin > tmax))
    return false;
  if (tymin > tmin)
    tmin = tymin;
  if (tymax < tmax)
    tmax = tymax;

  tzmin = (b[r.sign[2]].z - r.o.z) * r.invd.z;
  tzmax = (b[1 - r.sign[2]].z - r.o.z) * r.invd.z;

  if ((tmin > tzmax) || (tzmin > tmax))
    return false;
  if (tzmin > tmin)
    tmin = tzmin;
  if (tzmax < tmax)
    tmax = tzmax;

  return (tmin < tMax) && (tmax > tMin);
}

inline bool intersectAABB(const AABB &b, const Ray &r, double tMin, double tMax,
                          double &tEntry) {
  double tmin, tmax, tymin, tymax, tzmin, tzmax;

  tmin = (b[r.sign[0]].x - r.o.x) * r.invd.x;
  tmax = (b[1 - r.sign[0]].x - r.o.x) * r.invd.x;
  tymin = (b[r.sign[1]].y - r.o.y) * r.invd.y;
  tymax = (b[1 - r.sign[1]].y - r.o.y) * r.invd.y;

  if ((tmin > tymax) || (tymin > tmax))
    return false;
  if (tymin > tmin)
    tmin = tymin;
  if (tymax < tmax)
    tmax = tymax;

  tzmin = (b[r.sign[2]].z - r.o.z) * r.invd.z;
  tzmax = (b[1 - r.sign[2]].z - r.o.z) * r.invd.z;

  if ((tmin > tzmax) || (tzmin > tmax))
    return false;
  if (tzmin > tmin)
    tmin = tzmin;
  if (tzmax < tmax)
    tmax = tzmax;

  if ((tmin < tMax) && (tmax > tMin)) {
    tEntry = tmin;
    return true;
  }
  return false;
}

struct BVHNode {
  AABB box;
  std::unique_ptr<BVHNode> left, right;
  std::vector<int> triIndices, sphereIndices, planeIndices;
};

struct SAHBin {
  AABB bounds;
  int count = 0;
  void clear() {
    bounds = AABB();
    count = 0;
  }
};

std::unique_ptr<BVHNode>
buildBVH(const std::vector<Tri> &tris, const std::vector<Sphere> &spheres,
         const std::vector<Plane> &planes, const std::vector<int> &triIdx,
         const std::vector<int> &sphereIdx, const std::vector<int> &planeIdx,
         int depth = 0, int maxLeafSize = 4, int nBins = 8) {
  auto node = std::make_unique<BVHNode>();

  for (int i : triIdx) {
    AABB triBox;
    triBox.expand(tris[i].p0);
    triBox.expand(tris[i].p1);
    triBox.expand(tris[i].p2);
    node->box.expand(triBox);
  }

  for (int i : sphereIdx) {
    const Sphere &s = spheres[i];
    Vec3 r{s.radius, s.radius, s.radius};
    node->box.expand(s.pos - r);
    node->box.expand(s.pos + r);
  }

  for (int i : planeIdx) {
    const Plane &p = planes[i];
    Vec3 n = p.n;
    Vec3 any = std::abs(n.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 tangent = normalize(cross(n, any));
    Vec3 bitangent = normalize(cross(n, tangent));
    Vec3 center = -p.d * n;
    AABB ab;
    ab.expand(center - tangent * 1e5 - bitangent * 1e5 - n * 0.001);
    ab.expand(center + tangent * 1e5 + bitangent * 1e5 + n * 0.001);
    node->box.expand(ab);
  }

  int primCount =
      static_cast<int>(triIdx.size() + sphereIdx.size() + planeIdx.size());
  if (primCount <= maxLeafSize || depth > 24) {
    node->triIndices = triIdx;
    node->sphereIndices = sphereIdx;
    node->planeIndices = planeIdx;
    return node;
  }

  auto centroidTri = [&](int i) {
    return (tris[i].p0 + tris[i].p1 + tris[i].p2) * (1.0 / 3.0);
  };
  auto centroidSphere = [&](int i) { return spheres[i].pos; };
  auto centroidPlane = [&](int i) { return -planes[i].d * planes[i].n; };

  Vec3 cmin(INF, INF, INF), cmax(-INF, -INF, -INF);
  auto updateCentroid = [&](const Vec3 &c) {
    for (int k = 0; k < 3; ++k) {
      if (c[k] < cmin[k])
        cmin[k] = c[k];
      if (c[k] > cmax[k])
        cmax[k] = c[k];
    }
  };

  for (int i : triIdx)
    updateCentroid(centroidTri(i));
  for (int i : sphereIdx)
    updateCentroid(centroidSphere(i));
  for (int i : planeIdx)
    updateCentroid(centroidPlane(i));

  if (cmin.x >= cmax.x - 1e-12 && cmin.y >= cmax.y - 1e-12 &&
      cmin.z >= cmax.z - 1e-12) {
    node->triIndices = triIdx;
    node->sphereIndices = sphereIdx;
    node->planeIndices = planeIdx;
    return node;
  }

  Vec3 extent = cmax - cmin;
  int axis = 0;
  if (extent.y > extent.x && extent.y > extent.z)
    axis = 1;
  else if (extent.z > extent.x)
    axis = 2;

  nBins = std::max(2, nBins);
  std::vector<SAHBin> bins(nBins);
  for (auto &b : bins)
    b.clear();

  auto binIndex = [&](const Vec3 &c) {
    double v = (c[axis] - cmin[axis]) / (extent[axis] > 0 ? extent[axis] : 1.0);
    int idx = static_cast<int>(v * nBins);
    if (idx < 0)
      idx = 0;
    if (idx >= nBins)
      idx = nBins - 1;
    return idx;
  };

  for (int i : triIdx) {
    Vec3 c = centroidTri(i);
    int bi = binIndex(c);
    bins[bi].count++;
    AABB triBox;
    triBox.expand(tris[i].p0);
    triBox.expand(tris[i].p1);
    triBox.expand(tris[i].p2);
    bins[bi].bounds.expand(triBox);
  }
  for (int i : sphereIdx) {
    Vec3 c = centroidSphere(i);
    int bi = binIndex(c);
    bins[bi].count++;
    const Sphere &s = spheres[i];
    Vec3 r{s.radius, s.radius, s.radius};
    AABB sb;
    sb.expand(s.pos - r);
    sb.expand(s.pos + r);
    bins[bi].bounds.expand(sb);
  }
  for (int i : planeIdx) {
    Vec3 c = centroidPlane(i);
    int bi = binIndex(c);
    bins[bi].count++;
    const Plane &p = planes[i];
    Vec3 n = p.n;
    Vec3 any = std::abs(n.x) < 0.9 ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    Vec3 tangent = normalize(cross(n, any));
    Vec3 bitangent = normalize(cross(n, tangent));
    Vec3 center = -p.d * n;
    AABB ab;
    ab.expand(center - tangent * 1e5 - bitangent * 1e5 - n * 0.001);
    ab.expand(center + tangent * 1e5 + bitangent * 1e5 + n * 0.001);
    bins[bi].bounds.expand(ab);
  }

  std::vector<AABB> leftBounds(nBins), rightBounds(nBins + 1);
  std::vector<int> leftCounts(nBins), rightCounts(nBins + 1);

  AABB acc;
  int cnt = 0;
  for (int i = 0; i < nBins; ++i) {
    if (bins[i].count > 0) {
      if (cnt == 0)
        acc = bins[i].bounds;
      else
        acc.expand(bins[i].bounds);
    }
    cnt += bins[i].count;
    leftBounds[i] = acc;
    leftCounts[i] = cnt;
  }

  acc = AABB();
  cnt = 0;
  for (int i = nBins - 1; i >= 0; --i) {
    if (bins[i].count > 0) {
      if (cnt == 0)
        acc = bins[i].bounds;
      else
        acc.expand(bins[i].bounds);
    }
    cnt += bins[i].count;
    rightBounds[i + 1] = acc;
    rightCounts[i + 1] = cnt;
  }

  auto surfaceArea = [](const AABB &b) {
    Vec3 e = b.max - b.min;
    return 2.0 * (e.x * e.y + e.y * e.z + e.z * e.x);
  };
  double parentArea = std::max(1e-12, surfaceArea(node->box));

  double bestCost = INF;
  int bestBin = -1;
  for (int i = 0; i < nBins - 1; ++i) {
    int lc = leftCounts[i];
    int rc = rightCounts[i + 1];
    if (lc == 0 || rc == 0)
      continue;
    double leftArea = surfaceArea(leftBounds[i]);
    double rightArea = surfaceArea(rightBounds[i + 1]);
    double cost = (leftArea * lc + rightArea * rc) / parentArea;
    if (cost < bestCost) {
      bestCost = cost;
      bestBin = i;
    }
  }

  std::vector<int> leftTris, rightTris, leftSpheres, rightSpheres, leftPlanes,
      rightPlanes;
  if (bestBin == -1) {
    std::vector<std::pair<double, int>> centroids;
    centroids.reserve(primCount);
    for (int i : triIdx)
      centroids.emplace_back(centroidTri(i)[axis], i);
    for (int i : sphereIdx)
      centroids.emplace_back(centroidSphere(i)[axis], i + 1000000);
    for (int i : planeIdx)
      centroids.emplace_back(centroidPlane(i)[axis], i + 2000000);

    std::nth_element(centroids.begin(),
                     centroids.begin() + centroids.size() / 2, centroids.end(),
                     [](auto &a, auto &b) { return a.first < b.first; });

    double splitVal = centroids[centroids.size() / 2].first;

    for (int i : triIdx)
      (centroidTri(i)[axis] < splitVal ? leftTris : rightTris).push_back(i);
    for (int i : sphereIdx)
      (centroidSphere(i)[axis] < splitVal ? leftSpheres : rightSpheres)
          .push_back(i);
    for (int i : planeIdx)
      (centroidPlane(i)[axis] < splitVal ? leftPlanes : rightPlanes)
          .push_back(i);
  } else {
    double binWidth = extent[axis] / nBins;
    double splitCoord = cmin[axis] + binWidth * (bestBin + 1);
    for (int i : triIdx)
      (centroidTri(i)[axis] < splitCoord ? leftTris : rightTris).push_back(i);
    for (int i : sphereIdx)
      (centroidSphere(i)[axis] < splitCoord ? leftSpheres : rightSpheres)
          .push_back(i);
    for (int i : planeIdx)
      (centroidPlane(i)[axis] < splitCoord ? leftPlanes : rightPlanes)
          .push_back(i);

    if ((leftTris.empty() && leftSpheres.empty() && leftPlanes.empty()) ||
        (rightTris.empty() && rightSpheres.empty() && rightPlanes.empty())) {
      std::vector<std::pair<double, int>> centroids;
      centroids.reserve(primCount);
      for (int i : triIdx)
        centroids.emplace_back(centroidTri(i)[axis], i);
      for (int i : sphereIdx)
        centroids.emplace_back(centroidSphere(i)[axis], i + 1000000);
      for (int i : planeIdx)
        centroids.emplace_back(centroidPlane(i)[axis], i + 2000000);
      std::nth_element(
          centroids.begin(), centroids.begin() + centroids.size() / 2,
          centroids.end(), [](auto &a, auto &b) { return a.first < b.first; });
      double splitVal = centroids[centroids.size() / 2].first;
      leftTris.clear();
      rightTris.clear();
      leftSpheres.clear();
      rightSpheres.clear();
      leftPlanes.clear();
      rightPlanes.clear();
      for (int i : triIdx)
        (centroidTri(i)[axis] < splitVal ? leftTris : rightTris).push_back(i);
      for (int i : sphereIdx)
        (centroidSphere(i)[axis] < splitVal ? leftSpheres : rightSpheres)
            .push_back(i);
      for (int i : planeIdx)
        (centroidPlane(i)[axis] < splitVal ? leftPlanes : rightPlanes)
            .push_back(i);
    }
  }

  if (leftTris.empty() && leftSpheres.empty() && leftPlanes.empty()) {
    node->triIndices = triIdx;
    node->sphereIndices = sphereIdx;
    node->planeIndices = planeIdx;
    return node;
  }
  if (rightTris.empty() && rightSpheres.empty() && rightPlanes.empty()) {
    node->triIndices = triIdx;
    node->sphereIndices = sphereIdx;
    node->planeIndices = planeIdx;
    return node;
  }

  node->left = buildBVH(tris, spheres, planes, leftTris, leftSpheres,
                        leftPlanes, depth + 1, maxLeafSize, nBins);
  node->right = buildBVH(tris, spheres, planes, rightTris, rightSpheres,
                         rightPlanes, depth + 1, maxLeafSize, nBins);

  return node;
}

bool intersectBVH(const Ray &r, const std::unique_ptr<BVHNode> &root,
                  const std::vector<Tri> &tris,
                  const std::vector<Sphere> &spheres,
                  const std::vector<Plane> &planes, SceneHit &closest,
                  bool anyHit = false) {
  if (!root)
    return false;

  const BVHNode *stack[256];
  int sp = 0;

  double tEntry;
  if (!intersectAABB(root->box, r, EPS, closest.hit.t, tEntry))
    return false;

  stack[sp++] = root.get();
  bool hitSomething = false;

  while (sp) {
    const BVHNode *node = stack[--sp];

    if (closest.hit.t < INF) {
      if (!intersectAABB(node->box, r, EPS, closest.hit.t))
        continue;
    }

    double tMax = closest.hit.t;

    for (int i : node->triIndices) {
      Hit h;
      if (intersect_triangle(r, tris[i], h) && h.t >= EPS && h.t < tMax) {
        if (anyHit)
          return true;
        closest.kind = HitKind::Tri;
        closest.index = i;
        closest.hit = h;
        hitSomething = true;
        tMax = closest.hit.t;
      }
    }
    for (int i : node->sphereIndices) {
      Hit h;
      if (intersect_sphere(r, spheres[i], h) && h.t >= EPS && h.t < tMax) {
        if (anyHit)
          return true;
        closest.kind = HitKind::Sphere;
        closest.index = i;
        closest.hit = h;
        hitSomething = true;
        tMax = closest.hit.t;
      }
    }
    for (int i : node->planeIndices) {
      Hit h;
      if (intersect_plane(r, planes[i], h) && h.t >= EPS && h.t < tMax) {
        if (anyHit)
          return true;
        closest.kind = HitKind::Plane;
        closest.index = i;
        closest.hit = h;
        hitSomething = true;
        tMax = closest.hit.t;
      }
    }

    const BVHNode *L = node->left.get();
    const BVHNode *R = node->right.get();

    double tMinL, tMinR;
    bool hitL = L && intersectAABB(L->box, r, EPS, tMax, tMinL);
    bool hitR = R && intersectAABB(R->box, r, EPS, tMax, tMinR);

    if (hitL && hitR) {
      if (tMinL > tMinR) {
        stack[sp++] = L;
        stack[sp++] = R;
      } else {
        stack[sp++] = R;
        stack[sp++] = L;
      }
    } else if (hitL) {
      stack[sp++] = L;
    } else if (hitR) {
      stack[sp++] = R;
    }
  }

  return hitSomething;
}

bool is_blocked(const Ray &shadowRay, double maxT,
                const std::unique_ptr<BVHNode> &bvhRoot,
                const std::vector<Tri> &tris,
                const std::vector<Sphere> &spheres,
                const std::vector<Plane> &planes) {
  SceneHit closest;
  closest.hit.t = maxT;
  return intersectBVH(shadowRay, bvhRoot, tris, spheres, planes, closest, true);
}

RayTraceResult ray_trace(const Ray &ray, int depth, int bounces, int gi_depth,
                         int gi_max, const std::vector<Sphere> &spheres,
                         const std::vector<Light> &suns,
                         const std::vector<Light> &bulbs,
                         const std::vector<Plane> &planes,
                         const std::vector<std::vector<double>> &xyzs,
                         const std::vector<Tri> &tris,
                         const std::unique_ptr<BVHNode> &bvhRoot) {
  if (depth > bounces || gi_depth > gi_max)
    return {{0, 0, 0}, false};

  SceneHit closest;
  closest.kind = HitKind::None;
  closest.hit.t = INF;

  intersectBVH(ray, bvhRoot, tris, spheres, planes, closest);

  if (closest.kind == HitKind::None)
    return {{0, 0, 0}, false};

  Vec3 surfColor, normal, objShininess, objTransparency;
  double ior;

  if (closest.kind == HitKind::Sphere) {
    const Sphere &s = spheres[closest.index];
    normal = perturb_normal(normalize(closest.hit.position - s.pos), s.rough);
    if (s.tex) {
      double u =
          std::fmod(1.25 + std::atan2(normal.x, normal.z) / (2.0 * M_PI), 1.0);
      if (u < 0)
        u += 1.0;
      double v = 0.5 - std::asin(normal.y) / M_PI;
      surfColor = sample_texture(s.tex, u, v);
    } else {
      surfColor = s.color;
    }
    objShininess = s.shininess;
    objTransparency = s.transparency;
    ior = s.ior;
  } else if (closest.kind == HitKind::Plane) {
    const Plane &p = planes[closest.index];
    normal = perturb_normal(p.n, p.rough);
    surfColor = p.color;
    objShininess = p.shininess;
    objTransparency = p.transparency;
    ior = p.ior;
  } else {
    const Tri &t = tris[closest.index];
    normal = perturb_normal(t.normal, t.rough);
    if (t.tex) {
      Vec2 uv0{xyzs[t.i0][3], xyzs[t.i0][4]};
      Vec2 uv1{xyzs[t.i1][3], xyzs[t.i1][4]};
      Vec2 uv2{xyzs[t.i2][3], xyzs[t.i2][4]};
      Vec2 uv{closest.hit.b0 * uv0.u + closest.hit.b1 * uv1.u +
                  closest.hit.b2 * uv2.u,
              closest.hit.b0 * uv0.v + closest.hit.b1 * uv1.v +
                  closest.hit.b2 * uv2.v};
      surfColor = sample_texture(t.tex, uv.u, uv.v);
    } else {
      surfColor = t.color;
    }
    objShininess = t.shininess;
    objTransparency = t.transparency;
    ior = t.ior;
  }

  Vec3 I = normalize(ray.d);
  Vec3 n = normalize(normal);
  bool inside = false;
  if (dot(I, normal) > 0.0) {
    n = -1.0 * n;
    inside = true;
  }

  Vec3 col{0, 0, 0};

  for (const auto &s : suns) {
    Vec3 L = normalize(s.dir);
    Vec3 sunColor = s.color;
    double nDotL = dot(n, L);
    if (nDotL <= 0)
      continue;
    Ray shadowRay{closest.hit.position + L * EPS, L};
    if (!is_blocked(shadowRay, INF, bvhRoot, tris, spheres, planes))
      col = col + mulc(surfColor, sunColor) * nDotL;
  }

  for (const auto &b : bulbs) {
    Vec3 toBulb = b.dir - closest.hit.position;
    double dist2 = dot(toBulb, toBulb);
    if (dist2 < EPS * EPS)
      continue;
    double invDist = 1.0 / std::sqrt(dist2);
    Vec3 L = toBulb * invDist;
    double nDotL = dot(n, L);
    if (nDotL <= 0)
      continue;
    Ray shadowRay{closest.hit.position + L * EPS, L};
    if (!is_blocked(shadowRay, std::sqrt(dist2), bvhRoot, tris, spheres,
                    planes)) {
      Vec3 bulbColor = b.color * (1.0 / dist2);
      col = col + mulc(surfColor, bulbColor) * nDotL;
    }
  }

  if (gi_max > 0 && gi_depth <= gi_max) {
    Vec3 sphereCenter = closest.hit.position + n;

    double r = std::cbrt(randf());
    double theta = std::acos(randf() * 2.0 - 1);
    double phi = 2 * M_PI * randf();

    Vec3 rnd{std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
             std::cos(theta)};
    rnd = sphereCenter + rnd * r;

    Vec3 giDir = normalize(rnd - closest.hit.position);
    double nDotG = dot(n, giDir);
    if (nDotG > 0.0) {
      Ray giRay{closest.hit.position + giDir * EPS, giDir};
      RayTraceResult giRes =
          ray_trace(giRay, depth, bounces, gi_depth + 1, gi_max, spheres, suns,
                    bulbs, planes, xyzs, tris, bvhRoot);
      if (giRes.hit)
        col = col + mulc(surfColor, giRes.color) * nDotG;
    }
  }

  Vec3 reflCol{0, 0, 0};
  if (dot(objShininess, objShininess) > 0) {
    Vec3 refl = I - 2.0 * dot(n, I) * n;
    Ray reflected{closest.hit.position + refl * EPS, refl};
    RayTraceResult reflRes =
        ray_trace(reflected, depth + 1, bounces, gi_depth, gi_max, spheres,
                  suns, bulbs, planes, xyzs, tris, bvhRoot);
    reflCol = reflRes.color;
  }

  Vec3 refrCol{0, 0, 0};
  if (dot(objTransparency, objTransparency) > 0) {
    double eta = inside ? ior : 1.0 / ior;
    double cosi = dot(n, I);
    double k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    Vec3 refr =
        k < 0 ? I - 2.0 * cosi * n : eta * I - (eta * cosi + std::sqrt(k)) * n;
    Ray refracted{closest.hit.position + refr * EPS, refr};
    RayTraceResult refrRes =
        ray_trace(refracted, depth + 1, bounces, gi_depth, gi_max, spheres,
                  suns, bulbs, planes, xyzs, tris, bvhRoot);
    refrCol = refrRes.color;
  }

  static const Vec3 one = Vec3{1.0, 1.0, 1.0};
  Vec3 invSh = one - objShininess;
  Vec3 invTr = one - objTransparency;

  if (depth < bounces)
    col = mulc(col, mulc(invSh, invTr));

  return {col + mulc(reflCol, objShininess) +
              mulc(refrCol, mulc(invSh, objTransparency)),
          true};
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scenefile>\n";
    return 1;
  }
  const std::string sceneFile = argv[1];
  std::ifstream ifs(sceneFile);
  if (!ifs) {
    std::cerr << "Could not open " << sceneFile << "\n";
    return 1;
  }

  uint32_t width = 0, height = 0;
  std::string outFilename;
  int bounces = 4;
  Vec3 eye{0, 0, 0};
  Vec3 fwd{0, 0, -1};
  Vec3 rightVec{1, 0, 0};
  Vec3 up{0, 1, 0};
  double expose = 0.0;
  bool dof = false;
  double focus = 0.0;
  double lens = 0.0;
  int aa = 1;
  bool panorama = false;
  bool fisheye = false;
  int gi = 0;

  Vec3 curColor{1, 1, 1};
  Vec2 curTexCoord{0, 0};
  std::shared_ptr<TexData> curTex = nullptr;
  std::unordered_map<std::string, std::shared_ptr<TexData>> texCache;
  double roughness = 0.0;
  Vec3 shininess{0, 0, 0};
  Vec3 transparency{0, 0, 0};
  double ior = 1.458;

  std::vector<Sphere> spheres;
  std::vector<Light> suns;
  std::vector<Light> bulbs;
  std::vector<Plane> planes;
  std::vector<std::vector<double>> xyzs;
  std::vector<Tri> tris;

  std::string line;
  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.empty())
      continue;
    std::istringstream iss(line);
    std::string kw;
    iss >> kw;
    if (kw == "png") {
      iss >> width >> height >> outFilename;
    } else if (kw == "bounces") {
      iss >> bounces;
    } else if (kw == "forward") {
      iss >> fwd.x >> fwd.y >> fwd.z;
      rightVec = normalize(cross(fwd, up));
      up = normalize(cross(rightVec, fwd));
    } else if (kw == "up") {
      iss >> up.x >> up.y >> up.z;
      rightVec = normalize(cross(fwd, up));
      up = normalize(cross(rightVec, fwd));
    } else if (kw == "eye") {
      iss >> eye.x >> eye.y >> eye.z;
    } else if (kw == "expose") {
      iss >> expose;
    } else if (kw == "dof") {
      iss >> focus >> lens;
      dof = true;
    } else if (kw == "aa") {
      iss >> aa;
    } else if (kw == "panorama") {
      panorama = true;
    } else if (kw == "fisheye") {
      fisheye = true;
    } else if (kw == "gi") {
      iss >> gi;
    } else if (kw == "color") {
      iss >> curColor.x >> curColor.y >> curColor.z;
    } else if (kw == "texcoord") {
      iss >> curTexCoord.u >> curTexCoord.v;
    } else if (kw == "texture") {
      std::string texname;
      iss >> texname;
      if (texname == "none") {
        curTex = nullptr;
      } else {
        auto it = texCache.find(texname);
        if (it == texCache.end()) {
          auto img = std::make_shared<Image>(texname.c_str());
          auto td = std::make_shared<TexData>();
          td->w = img->width();
          td->h = img->height();
          td->linear.resize(td->w * td->h);
          for (int yy = 0; yy < td->h; ++yy) {
            for (int xx = 0; xx < td->w; ++xx) {
              const auto &px = (*img)[yy][xx];
              td->linear[yy * td->w + xx] = {sRGBToLinear(px.r / 255.0),
                                             sRGBToLinear(px.g / 255.0),
                                             sRGBToLinear(px.b / 255.0)};
            }
          }
          texCache[texname] = td;
          curTex = td;
        } else {
          curTex = it->second;
        }
      }
    } else if (kw == "roughness") {
      iss >> roughness;
    } else if (kw == "shininess") {
      iss >> shininess.x;
      if (iss >> shininess.y)
        iss >> shininess.z;
      else
        shininess.z = shininess.y = shininess.x;
    } else if (kw == "transparency") {
      iss >> transparency.x;
      if (iss >> transparency.y)
        iss >> transparency.z;
      else
        transparency.z = transparency.y = transparency.x;
    } else if (kw == "ior") {
      iss >> ior;
    } else if (kw == "sphere") {
      Sphere s;
      iss >> s.pos.x >> s.pos.y >> s.pos.z;
      iss >> s.radius;
      s.color = curColor;
      s.tex = curTex;
      s.rough = roughness;
      s.shininess = shininess;
      s.transparency = transparency;
      s.ior = ior;
      spheres.push_back(std::move(s));
    } else if (kw == "sun") {
      Light s;
      iss >> s.dir[0] >> s.dir[1] >> s.dir[2];
      s.color = curColor;
      suns.push_back(std::move(s));
    } else if (kw == "bulb") {
      Light b;
      iss >> b.dir[0] >> b.dir[1] >> b.dir[2];
      b.color = curColor;
      bulbs.push_back(std::move(b));
    } else if (kw == "plane") {
      Plane p;
      iss >> p.n.x >> p.n.y >> p.n.z >> p.d;
      p.color = curColor;
      p.rough = roughness;
      p.shininess = shininess;
      p.transparency = transparency;
      p.ior = ior;
      planes.push_back(std::move(p));
    } else if (kw == "xyz") {
      std::vector<double> xyz(5);
      iss >> xyz[0] >> xyz[1] >> xyz[2];
      xyz[3] = curTexCoord.u;
      xyz[4] = curTexCoord.v;
      xyzs.push_back(std::move(xyz));
    } else if (kw == "tri") {
      int i0, i1, i2;
      iss >> i0 >> i1 >> i2;

      auto idxFix = [&xyzs](int idx) -> int {
        if (idx > 0)
          return idx - 1;
        return static_cast<int>(xyzs.size()) + idx;
      };

      Tri t;
      t.i0 = idxFix(i0);
      t.i1 = idxFix(i1);
      t.i2 = idxFix(i2);
      t.p0 = Vec3(xyzs[t.i0][0], xyzs[t.i0][1], xyzs[t.i0][2]);
      t.p1 = Vec3(xyzs[t.i1][0], xyzs[t.i1][1], xyzs[t.i1][2]);
      t.p2 = Vec3(xyzs[t.i2][0], xyzs[t.i2][1], xyzs[t.i2][2]);
      t.color = curColor;
      t.tex = curTex;
      t.rough = roughness;
      t.shininess = shininess;
      t.transparency = transparency;
      t.ior = ior;
      t.compute_pre();
      tris.push_back(std::move(t));
    }
  }

  std::vector<int> triIndices(tris.size()), sphereIndices(spheres.size()),
      planeIndices(planes.size());
  std::iota(triIndices.begin(), triIndices.end(), 0);
  std::iota(sphereIndices.begin(), sphereIndices.end(), 0);
  std::iota(planeIndices.begin(), planeIndices.end(), 0);
  auto bvhRoot =
      buildBVH(tris, spheres, planes, triIndices, sphereIndices, planeIndices);

  auto img = std::make_unique<Image>(width, height);

  double denom = static_cast<double>(std::max(width, height));

  omp_set_num_threads(16);
#pragma omp parallel for collapse(2) schedule(dynamic)
  for (uint32_t ty = 0; ty < height; ty += 16) {
    for (uint32_t tx = 0; tx < width; tx += 16) {
      for (uint32_t y = ty; y < std::min(ty + 16, height); ++y) {
        for (uint32_t x = tx; x < std::min(tx + 16, width); ++x) {
          Vec3 colSum{0, 0, 0};
          double alphaSum = 0.0;

          for (int s = 0; s < aa; ++s) {
            double dx = s == 0 ? 0.0 : randf() - 0.5;
            double dy = s == 0 ? 0.0 : randf() - 0.5;

            Vec3 dir;
            if (panorama) {
              double u = 1.0 - (x + dx) / static_cast<double>(width);
              double v = 1.0 - (y + dy) / static_cast<double>(height);
              double lon = u * 2.0 * M_PI - M_PI / 2.0;
              double lat = v * M_PI - M_PI / 2.0;
              double cosLat = std::cos(lat);
              double sinLat = std::sin(lat);
              double cosLon = std::cos(lon);
              double sinLon = std::sin(lon);
              dir = normalize({cosLat * cosLon, sinLat, -cosLat * sinLon});
            } else if (fisheye) {
              double sx = (2.0 * (x + dx) - static_cast<double>(width)) / denom;
              double sy =
                  (static_cast<double>(height) - 2.0 * (y + dy)) / denom;
              double r2 = sx * sx + sy * sy;
              if (r2 > 1.0)
                continue;
              dir = normalize(fwd * std::sqrt(1.0 - r2) + rightVec * sx +
                              up * sy);
            } else {
              double sx = (2.0 * (x + dx) - static_cast<double>(width)) / denom;
              double sy =
                  (static_cast<double>(height) - 2.0 * (y + dy)) / denom;
              dir = normalize(fwd + rightVec * sx + up * sy);
            }

            Vec3 ro = eye;
            Vec3 rd = dir;

            if (dof && !fisheye && !panorama && lens > 0.0 && focus > 0.0) {
              double r = std::sqrt(randf());
              double theta = 2.0 * M_PI * randf();
              double dxLens = r * std::cos(theta);
              double dyLens = r * std::sin(theta);

              Vec3 lensOffset =
                  rightVec * (dxLens * lens) + up * (dyLens * lens);
              Vec3 o_new = eye + lensOffset;
              Vec3 focalPoint = eye + dir * focus;
              rd = normalize(focalPoint - o_new);
              ro = o_new;
            }

            Ray ray{ro, rd};
            RayTraceResult res =
                ray_trace(ray, 0, bounces, 0, gi, spheres, suns, bulbs, planes,
                          xyzs, tris, bvhRoot);
            colSum = colSum + res.color;
            alphaSum += res.hit;
          }

          Vec3 col = alphaSum > 0 ? colSum * (1.0 / alphaSum) : Vec3{0, 0, 0};
          double alpha = alphaSum / aa;

          if (expose) {
            col.x = 1.0 - std::exp(-expose * col.x);
            col.y = 1.0 - std::exp(-expose * col.y);
            col.z = 1.0 - std::exp(-expose * col.z);
          }

          double r = linearToSRGB(clamp01(col.x));
          double g = linearToSRGB(clamp01(col.y));
          double b = linearToSRGB(clamp01(col.z));

          (*img)[y][x].r = static_cast<uint8_t>(r * 255.0);
          (*img)[y][x].g = static_cast<uint8_t>(g * 255.0);
          (*img)[y][x].b = static_cast<uint8_t>(b * 255.0);
          (*img)[y][x].a = static_cast<uint8_t>(alpha * 255.0);
        }
      }
    }
  }

  img->save(outFilename.c_str());
  return 0;
}
