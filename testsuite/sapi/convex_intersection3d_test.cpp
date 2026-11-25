/*
 *  convex_intersection3d_test.cpp
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

// The 3D GJK path (convex_intersection.h with Coord3D), on top of the sphere
// and axis-aligned box checks of convex_intersection_test.cpp:
//   test_rotation            create_rotation_3D, rotate_displacement and
//                            inverse_rotate_displacement against explicit
//                            matrix products; supports of rotated boxes
//   test_exact_boxes         boxes rotated by multiples of 90 degrees touching
//                            in every contact configuration, exact in doubles
//   test_hull_supports       hulls of rotated boxes, tetrahedra, triangular and
//                            hexagonal prisms, segments and triangles, and
//                            spheres, against the exact distance of an
//                            independent oracle (separating axes of all
//                            vertex/edge/face pairs, long double)
//   test_contact_certification  pairs at an exact gap g touching in every
//                            contact type (vertex-vertex ... face-face), far
//                            from the origin: GJK must certify 4 delta gaps
//                            at margin delta and report touching, delta / 2
//                            gaps and overlaps
//   test_simplex_near_touching  closest_point_on_simplex_to_origin when the
//                            closest point is tiny against the vertices,
//                            against a Gaussian-elimination oracle
//   test_random_polytopes    rotated polytopes against the separating-axis
//                            oracle, also with margins close to the exact gap

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "convex_intersection.h"
#include "coordinate_geometry.h"
#include "geometry_oracle.h"
#include "sapi_config.h"

using namespace sapi;
using geometry_test::fail;
using geometry_test::require;

namespace {
using Rng = std::mt19937_64;
using LD = long double;
using V = std::array<LD, 3>;
using Rows3 = std::array<V, 3>;
using Cloud = std::vector<V>;

// A direction m of a local polytope along which a feature of the given rank
// (0 vertex, 1 edge, 2 face) is farthest, and for an edge its unit direction
// e (orthogonal to m)
struct FeatureEntry {
  V m;
  V e;
};

using FeatureTable = std::array<std::vector<FeatureEntry>, 3>;

constexpr LD PI = 3.14159265358979323846264338327950288L;

double uniform(Rng &rng, const double lo, const double hi) {
  return std::uniform_real_distribution<double>(lo, hi)(rng);
}

int uniform_int(Rng &rng, const int lo, const int hi) {
  return std::uniform_int_distribution<int>(lo, hi)(rng);
}

std::size_t pick(Rng &rng, const std::size_t count) {
  return static_cast<std::size_t>(rng() % count);
}

// ---------------------------------------------------------------------------
// Long double vector algebra of the oracles (independent of sapi::Coord3D)
// ---------------------------------------------------------------------------
V operator+(const V &a, const V &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

V operator-(const V &a, const V &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

V operator*(const V &a, const LD s) { return {a[0] * s, a[1] * s, a[2] * s}; }

LD vdot(const V &a, const V &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

V vcross(const V &a, const V &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

LD vnorm2(const V &a) { return vdot(a, a); }

LD vnorm(const V &a) { return std::sqrt(vnorm2(a)); }

V vunit(const V &a) { return a * (1 / vnorm(a)); }

V to_v(const Coord3D &c) { return {c.x_, c.y_, c.z_}; }

Coord3D to_c3(const V &v) {
  return construct_coord_3D(static_cast<double>(v[0]),
                            static_cast<double>(v[1]),
                            static_cast<double>(v[2]));
}

Cloud to_cloud(const std::vector<Coord3D> &points) {
  Cloud cloud;
  for (const auto &p : points)
    cloud.push_back(to_v(p));
  return cloud;
}

V random_unit(Rng &rng) {
  std::normal_distribution<double> gauss;
  for (;;) {
    const double x = gauss(rng);
    const double y = gauss(rng);
    const double z = gauss(rng);
    const V v{x, y, z};
    if (1e-3L < vnorm(v))
      return vunit(v);
  }
}

// Random unit vector orthogonal to the unit vector n
V perpendicular_unit(const V &n, Rng &rng) {
  for (;;) {
    const V r = random_unit(rng);
    const V t = r - n * vdot(r, n);
    if (1e-4L < vnorm2(t))
      return vunit(t);
  }
}

Coord3D random_point(Rng &rng, const double reach) {
  return construct_coord_3D(uniform(rng, -reach, reach),
                            uniform(rng, -reach, reach),
                            uniform(rng, -reach, reach));
}

bool finite(const Coord3D &c) {
  return std::isfinite(c.x_) && std::isfinite(c.y_) && std::isfinite(c.z_);
}

// ---------------------------------------------------------------------------
// Rotations: explicit matrix products as reference
// ---------------------------------------------------------------------------

// cos and sin of an integer angle in degrees, exact at multiples of 90
LD cos_degrees(const int degrees) {
  if (degrees % 90 == 0) {
    constexpr std::array<LD, 4> table{1.L, 0.L, -1.L, 0.L};
    return table[static_cast<std::size_t>(((degrees / 90) % 4 + 4) % 4)];
  }
  return std::cos(static_cast<LD>(degrees) * PI / 180);
}

LD sin_degrees(const int degrees) {
  if (degrees % 90 == 0) {
    constexpr std::array<LD, 4> table{0.L, 1.L, 0.L, -1.L};
    return table[static_cast<std::size_t>(((degrees / 90) % 4 + 4) % 4)];
  }
  return std::sin(static_cast<LD>(degrees) * PI / 180);
}

Rows3 matrix_product(const Rows3 &a, const Rows3 &b) {
  Rows3 product{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        product[i][j] += a[i][k] * b[k][j];
  return product;
}

V matrix_apply(const Rows3 &m, const V &p) {
  return {vdot(m[0], p), vdot(m[1], p), vdot(m[2], p)};
}

// R^T p
V transposed_apply(const Rows3 &m, const V &p) {
  return m[0] * p[0] + m[1] * p[1] + m[2] * p[2];
}

// Rz(alpha) Ry(beta) Rx(gamma) as the product of the three plain rotations
Rows3 reference_rotation(const int alpha, const int beta, const int gamma) {
  const LD ca = cos_degrees(alpha), sa = sin_degrees(alpha);
  const LD cb = cos_degrees(beta), sb = sin_degrees(beta);
  const LD cg = cos_degrees(gamma), sg = sin_degrees(gamma);
  const Rows3 rz{V{ca, -sa, 0.L}, V{sa, ca, 0.L}, V{0.L, 0.L, 1.L}};
  const Rows3 ry{V{cb, 0.L, sb}, V{0.L, 1.L, 0.L}, V{-sb, 0.L, cb}};
  const Rows3 rx{V{1.L, 0.L, 0.L}, V{0.L, cg, -sg}, V{0.L, sg, cg}};
  return matrix_product(rz, matrix_product(ry, rx));
}

// Rotation mapping the orthonormal pair (m, e) onto (big_n, t), a proper
// rotation: R = sum_k world_k local_k^T with the third axes the cross products
Rows3 frame_rotation(const V &m, const V &e, const V &big_n, const V &t) {
  const std::array<V, 3> local{m, e, vcross(m, e)};
  const std::array<V, 3> world{big_n, t, vcross(big_n, t)};
  Rows3 r{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t k = 0; k < 3; ++k)
        r[i][j] += world[k][i] * local[k][j];
  return r;
}

// A multiple of 90 degrees a quarter of the times
angle_t random_angle(Rng &rng) {
  if (rng() % 4 == 0)
    return static_cast<angle_t>(90 * uniform_int(rng, -4, 4));
  return static_cast<angle_t>(uniform_int(rng, -360, 360));
}

Rotation3D random_rotation(Rng &rng) {
  const angle_t alpha = random_angle(rng);
  const angle_t beta = random_angle(rng);
  const angle_t gamma = random_angle(rng);
  return create_rotation_3D(alpha, beta, gamma);
}

// ---------------------------------------------------------------------------
// Convex sets: hull of points (support function) and ball
// ---------------------------------------------------------------------------
struct Hull {
  std::vector<Coord3D> points;

  Coord3D operator()(const Coord3D &u) const {
    std::size_t best = 0;
    double best_dot = vector_dot(points[0], u);
    for (std::size_t i = 1; i < points.size(); ++i) {
      const double d = vector_dot(points[i], u);
      if (best_dot < d) {
        best_dot = d;
        best = i;
      }
    }
    return points[best];
  }
};

struct Ball {
  Coord3D center;
  double radius = 0.;

  Coord3D operator()(const Coord3D &u) const {
    const double n = vector_norm(u);
    return 0 < n ? center + u * (radius / n) : center;
  }
};

// Mean of the points and the largest distance to it: the bounding sphere the
// margin is sized by (exact_filter_margin takes the bounding disks)
Ball bounding_ball(const std::vector<Coord3D> &points) {
  Coord3D sum;
  for (const auto &p : points)
    sum = sum + p;
  Ball ball{sum / static_cast<double>(points.size()), 0.};
  for (const auto &p : points)
    ball.radius = std::fmax(ball.radius, distance(p, ball.center));
  return ball;
}

// Absolute margin in the style of exact_filter_margin of two bounding spheres
double filter_margin(const Ball &a, const Ball &b) {
  return EXACT_FILTER_REL_TOL *
         std::sqrt(a.radius * a.radius + vector_norm2(a.center) +
                   b.radius * b.radius + vector_norm2(b.center));
}

// Bodies, each a convex set of a polytope kind or a ball
enum class PolyKind {
  BOX,
  TETRAHEDRON,
  TRIANGULAR_PRISM,
  HEXAGONAL_PRISM,
  SEGMENT,
  TRIANGLE
};

constexpr std::array<PolyKind, 6> POLY_KINDS{PolyKind::BOX,
                                             PolyKind::TETRAHEDRON,
                                             PolyKind::TRIANGULAR_PRISM,
                                             PolyKind::HEXAGONAL_PRISM,
                                             PolyKind::SEGMENT,
                                             PolyKind::TRIANGLE};

std::string kind_name(const PolyKind kind) {
  switch (kind) {
  case PolyKind::BOX:
    return "box";
  case PolyKind::TETRAHEDRON:
    return "tetrahedron";
  case PolyKind::TRIANGULAR_PRISM:
    return "triangular prism";
  case PolyKind::HEXAGONAL_PRISM:
    return "hexagonal prism";
  case PolyKind::SEGMENT:
    return "segment";
  default:
    return "triangle";
  }
}

Cloud prism_vertices(const int sides, const LD half_height) {
  Cloud vertices;
  for (int s = 0; s < sides; ++s) {
    const LD angle = PI / 2 + 2 * PI * static_cast<LD>(s) / sides;
    for (const LD z : {-half_height, half_height})
      vertices.push_back({std::cos(angle), std::sin(angle), z});
  }
  return vertices;
}

// Vertices of a polytope of the kind, about unit size, centered on their mean
Cloud local_polytope(const PolyKind kind, Rng &rng) {
  Cloud p;
  switch (kind) {
  case PolyKind::BOX: {
    const LD hx = uniform(rng, 0.3, 1.), hy = uniform(rng, 0.3, 1.),
             hz = uniform(rng, 0.3, 1.);
    for (const LD x : {-hx, hx})
      for (const LD y : {-hy, hy})
        for (const LD z : {-hz, hz})
          p.push_back({x, y, z});
    break;
  }
  case PolyKind::TETRAHEDRON:
    for (;;) {
      p.clear();
      for (int i = 0; i < 4; ++i)
        p.push_back(to_v(random_point(rng, 1.)));
      if (0.3L < std::fabs(vdot(p[1] - p[0], vcross(p[2] - p[0], p[3] - p[0]))))
        break;
    }
    break;
  case PolyKind::TRIANGULAR_PRISM:
    p = prism_vertices(3, uniform(rng, 0.3, 1.));
    break;
  case PolyKind::HEXAGONAL_PRISM:
    p = prism_vertices(6, uniform(rng, 0.3, 1.));
    break;
  case PolyKind::SEGMENT:
    for (;;) {
      p.clear();
      for (int i = 0; i < 2; ++i)
        p.push_back(to_v(random_point(rng, 1.)));
      if (0.5L < vnorm(p[1] - p[0]))
        break;
    }
    break;
  default:
    for (;;) {
      p.clear();
      for (int i = 0; i < 3; ++i)
        p.push_back(to_v(random_point(rng, 1.)));
      if (0.3L < vnorm(vcross(p[1] - p[0], p[2] - p[0])))
        break;
    }
    break;
  }

  V mean{};
  for (const auto &q : p)
    mean = mean + q;
  mean = mean * (1 / static_cast<LD>(p.size()));
  for (auto &q : p)
    q = q - mean;
  return p;
}

// A body of the tests: points of a polytope (hull) or a ball, with the
// quantities the margin and the query direction need
struct TestBody {
  bool round = false;
  std::string label;
  Hull hull;
  Ball ball;
  Cloud exact; // the hull points in long double, for the oracles
};

TestBody polytope_body(const PolyKind kind, Rng &rng, const double scale,
                       const Coord3D &offset) {
  const Cloud local = local_polytope(kind, rng);
  const Rotation3D rotation = random_rotation(rng);
  TestBody body;
  body.label = kind_name(kind);
  for (const auto &p : local)
    body.hull.points.push_back(
        offset + inverse_rotate_displacement(to_c3(p) * scale, rotation));
  body.ball = bounding_ball(body.hull.points);
  body.exact = to_cloud(body.hull.points);
  return body;
}

TestBody ball_body(const Coord3D &center, const double radius) {
  TestBody body;
  body.round = true;
  body.label = "ball";
  body.ball = {center, radius};
  return body;
}

bool bodies_meet(const TestBody &a, const TestBody &b, const double margin) {
  const Coord3D direction = a.ball.center - b.ball.center;
  if (a.round && b.round)
    return convex_sets_intersect(a.ball, b.ball, direction, margin);
  if (a.round)
    return convex_sets_intersect(a.ball, b.hull, direction, margin);
  if (b.round)
    return convex_sets_intersect(a.hull, b.ball, direction, margin);
  return convex_sets_intersect(a.hull, b.hull, direction, margin);
}

// ---------------------------------------------------------------------------
// Separation oracle for convex polytopes (also points, segments, triangles).
// The distance of two disjoint polytopes is the gap along the direction of
// their closest pair of points, and that direction is, by the type of the
// features holding the pair, one of:
//   vertex-vertex  the difference of the vertices
//   vertex-edge    the difference orthogonalized against the edge
//   vertex-face    a face normal (also for edge-face and face-face)
//   edge-edge      the cross product of the edges
// Every candidate axis gives a valid lower bound (a separating axis), so the
// largest gap over all candidates built from all vertex pairs, triples and
// differences is the distance, and it is not positive iff the sets meet.
// ---------------------------------------------------------------------------
struct Extent {
  LD lo, hi;
};

Extent project(const Cloud &points, const V &axis) {
  Extent extent{std::numeric_limits<LD>::infinity(),
                -std::numeric_limits<LD>::infinity()};
  for (const auto &p : points) {
    const LD d = vdot(p, axis);
    extent.lo = std::fmin(extent.lo, d);
    extent.hi = std::fmax(extent.hi, d);
  }
  return extent;
}

// Gap of a and b along the raw axis, ignored when the axis is negligible
// against the product of the factors' squared lengths it was built from
void consider_axis(const Cloud &a, const Cloud &b, const V &axis,
                   const LD factor2, LD &best) {
  const LD n2 = vnorm2(axis);
  if (!(1e-24L * factor2 < n2))
    return;
  const V unit = axis * (1 / std::sqrt(n2));
  const Extent ea = project(a, unit);
  const Extent eb = project(b, unit);
  best = std::fmax(best, std::fmax(eb.lo - ea.hi, ea.lo - eb.hi));
}

void consider_faces(const Cloud &a, const Cloud &b, const Cloud &body,
                    LD &best) {
  for (std::size_t i = 0; i < body.size(); ++i)
    for (std::size_t j = i + 1; j < body.size(); ++j)
      for (std::size_t k = j + 1; k < body.size(); ++k) {
        const V u = body[j] - body[i];
        const V w = body[k] - body[i];
        consider_axis(a, b, vcross(u, w), vnorm2(u) * vnorm2(w), best);
      }
}

void consider_vertex_edges(const Cloud &a, const Cloud &b,
                           const Cloud &vertices, const Cloud &body, LD &best) {
  for (const auto &p : vertices)
    for (std::size_t i = 0; i < body.size(); ++i)
      for (std::size_t j = i + 1; j < body.size(); ++j) {
        const V e = body[j] - body[i];
        const LD e2 = vnorm2(e);
        if (!(0 < e2))
          continue;
        const V d = body[i] - p;
        consider_axis(a, b, d - e * (vdot(d, e) / e2), vnorm2(d), best);
      }
}

LD separation(const Cloud &a, const Cloud &b) {
  LD best = -std::numeric_limits<LD>::infinity();
  consider_faces(a, b, a, best);
  consider_faces(a, b, b, best);
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = i + 1; j < a.size(); ++j)
      for (std::size_t k = 0; k < b.size(); ++k)
        for (std::size_t l = k + 1; l < b.size(); ++l) {
          const V e = a[j] - a[i];
          const V f = b[l] - b[k];
          consider_axis(a, b, vcross(e, f), vnorm2(e) * vnorm2(f), best);
        }
  for (const auto &p : a)
    for (const auto &q : b)
      consider_axis(a, b, q - p, vnorm2(p) + vnorm2(q), best);
  consider_vertex_edges(a, b, a, b, best);
  consider_vertex_edges(a, b, b, a, best);
  return best;
}

// Distance (negative inside, bounded below by the oracle's own gap) of two
// bodies
LD body_gap(const TestBody &a, const TestBody &b) {
  if (a.round && b.round)
    return vnorm(to_v(a.ball.center) - to_v(b.ball.center)) -
           static_cast<LD>(a.ball.radius) - static_cast<LD>(b.ball.radius);
  if (a.round)
    return separation(b.exact, Cloud{to_v(a.ball.center)}) -
           static_cast<LD>(a.ball.radius);
  if (b.round)
    return separation(a.exact, Cloud{to_v(b.ball.center)}) -
           static_cast<LD>(b.ball.radius);
  return separation(a.exact, b.exact);
}

// Size of the pair for the tolerances: sqrt of the squared radii and centers
// as in the margin
double reference_size(const TestBody &a, const TestBody &b) {
  return std::sqrt(a.ball.radius * a.ball.radius + vector_norm2(a.ball.center) +
                   b.ball.radius * b.ball.radius + vector_norm2(b.ball.center));
}

// ---------------------------------------------------------------------------
// 1. Rotations and supports of rotated boxes
// ---------------------------------------------------------------------------
void test_rotation() {
  Rng rng(3301);
  for (int k = 0; k < 3000; ++k) {
    const bool quarter_turns = k % 3 == 0;
    std::array<int, 3> degrees{};
    for (auto &a : degrees)
      a = quarter_turns ? 90 * uniform_int(rng, -4, 4)
                        : uniform_int(rng, -360, 360);
    const Rotation3D rotation = create_rotation_3D(
        static_cast<angle_t>(degrees[0]), static_cast<angle_t>(degrees[1]),
        static_cast<angle_t>(degrees[2]));
    const Rows3 reference =
        reference_rotation(degrees[0], degrees[1], degrees[2]);
    const std::string where = "rotation " + std::to_string(degrees[0]) + " " +
                              std::to_string(degrees[1]) + " " +
                              std::to_string(degrees[2]);

    Rows3 rows;
    for (std::size_t i = 0; i < 3; ++i)
      rows[i] = to_v(rotation[i]);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j) {
        if (quarter_turns)
          require(rows[i][j] == reference[i][j],
                  "multiples of 90 degrees are not exact: " + where);
        else
          require(std::fabs(rows[i][j] - reference[i][j]) <= 2e-15L,
                  "rotation entry: " + where);
        require(std::fabs(vdot(rows[i], rows[j]) - (i == j ? 1.L : 0.L)) <=
                    2e-15L,
                "rotation is not orthonormal: " + where);
      }
    require(std::fabs(vdot(rows[0], vcross(rows[1], rows[2])) - 1) <= 2e-15L,
            "rotation is not proper: " + where);

    const Coord3D d = random_point(rng, 10.);
    const V forward = transposed_apply(reference, to_v(d));
    const V inverse = matrix_apply(reference, to_v(d));
    require(vnorm(to_v(rotate_displacement(d, rotation)) - forward) <= 1e-13L,
            "rotate_displacement is not R^T d: " + where);
    require(vnorm(to_v(inverse_rotate_displacement(d, rotation)) - inverse) <=
                1e-13L,
            "inverse_rotate_displacement is not R d: " + where);
    require(vnorm(to_v(rotate_displacement(
                      inverse_rotate_displacement(d, rotation), rotation)) -
                  to_v(d)) <= 1e-13L,
            "rotations are not inverse: " + where);
  }

  // A rotation about z alone is the 2D rotation, z kept
  for (int alpha = -360; alpha <= 360; alpha += 7) {
    const Rotation3D rotation =
        create_rotation_3D(static_cast<angle_t>(alpha), 0, 0);
    const Coord3D d = random_point(rng, 10.);
    const Coord3D got = rotate_displacement(d, rotation);
    const Coord2D flat =
        rotate_displacement(construct_coord_2D(d.x_, d.y_),
                            create_angular_offset(static_cast<double>(alpha)));
    require(std::fabs(got.x_ - flat.x_) <= 1e-13 &&
                std::fabs(got.y_ - flat.y_) <= 1e-13 &&
                std::fabs(got.z_ - d.z_) <= 1e-15,
            "z rotation differs from the 2D rotation: " +
                std::to_string(alpha));
  }

  // Support of a rotated box: the local support along R^T u, rotated back
  for (int k = 0; k < 500; ++k) {
    const std::array<double, 3> half{
        uniform(rng, 0.2, 2.), uniform(rng, 0.2, 2.), uniform(rng, 0.2, 2.)};
    const std::array<int, 3> degrees{uniform_int(rng, -360, 360),
                                     uniform_int(rng, -360, 360),
                                     uniform_int(rng, -360, 360)};
    const Rotation3D rotation = create_rotation_3D(
        static_cast<angle_t>(degrees[0]), static_cast<angle_t>(degrees[1]),
        static_cast<angle_t>(degrees[2]));
    const Rows3 reference =
        reference_rotation(degrees[0], degrees[1], degrees[2]);
    Hull box;
    for (const double sx : {-1., 1.})
      for (const double sy : {-1., 1.})
        for (const double sz : {-1., 1.})
          box.points.push_back(inverse_rotate_displacement(
              construct_coord_3D(sx * half[0], sy * half[1], sz * half[2]),
              rotation));
    for (int i = 0; i < 20; ++i) {
      const V u = random_unit(rng);
      const V local = transposed_apply(reference, u);
      const V corner{local[0] < 0 ? -half[0] : half[0],
                     local[1] < 0 ? -half[1] : half[1],
                     local[2] < 0 ? -half[2] : half[2]};
      const LD expected = vdot(matrix_apply(reference, corner), u);
      const LD got = vdot(to_v(box(to_c3(u))), u);
      require(std::fabs(got - expected) <= 1e-12L, "support of a rotated box");
    }
  }
}

// ---------------------------------------------------------------------------
// 2. Boxes rotated by multiples of 90 degrees: exact touching
// ---------------------------------------------------------------------------
std::vector<Coord3D> quarter_turn_box(const Rotation3D &rotation,
                                      const Coord3D &offset) {
  std::vector<Coord3D> corners;
  for (const double sx : {-0.5, 0.5})
    for (const double sy : {-1., 1.})
      for (const double sz : {-1.5, 1.5})
        corners.push_back(
            offset + inverse_rotate_displacement(construct_coord_3D(sx, sy, sz),
                                                 rotation));
  return corners;
}

Coord3D half_extents(const std::vector<Coord3D> &corners) {
  Coord3D extent;
  for (const auto &c : corners) {
    extent.x_ = std::fmax(extent.x_, std::fabs(c.x_));
    extent.y_ = std::fmax(extent.y_, std::fabs(c.y_));
    extent.z_ = std::fmax(extent.z_, std::fabs(c.z_));
  }
  return extent;
}

void test_exact_boxes() {
  Rng rng(4242);
  constexpr std::array<int, 4> turns{0, 90, 180, 270};
  const auto rotation_of = [&](const std::size_t index) {
    return create_rotation_3D(static_cast<angle_t>(turns[index / 16]),
                              static_cast<angle_t>(turns[index / 4 % 4]),
                              static_cast<angle_t>(turns[index % 4]));
  };
  long checked = 0;
  for (std::size_t ia = 0; ia < 64; ++ia) {
    const Hull a{quarter_turn_box(rotation_of(ia), Coord3D())};
    const Coord3D extent_a = half_extents(a.points);
    for (int pick_b = 0; pick_b < 4; ++pick_b) {
      const Rotation3D rotation_b = rotation_of(pick(rng, 64));
      const Coord3D extent_b =
          half_extents(quarter_turn_box(rotation_b, Coord3D()));
      for (int sx = -1; sx <= 1; ++sx)
        for (int sy = -1; sy <= 1; ++sy)
          for (int sz = -1; sz <= 1; ++sz) {
            if (sx == 0 && sy == 0 && sz == 0)
              continue;
            const Coord3D contact =
                construct_coord_3D(sx * (extent_a.x_ + extent_b.x_),
                                   sy * (extent_a.y_ + extent_b.y_),
                                   sz * (extent_a.z_ + extent_b.z_));
            const Coord3D slack =
                construct_coord_3D(sx * 1e-3, sy * 1e-3, sz * 1e-3);
            const auto meets = [&](const Coord3D &offset, const double margin) {
              const Hull b{quarter_turn_box(rotation_b, offset)};
              return convex_sets_intersect(a, b, offset.negative(), margin);
            };
            const std::string where = "boxes " + std::to_string(ia) + " " +
                                      std::to_string(sx) + std::to_string(sy) +
                                      std::to_string(sz);
            // Touching and overlapping with a margin far below the gaps: at
            // margin 0 a rounding of the closest point may certify a gap
            require(meets(contact, 1e-9), "exact touching boxes: " + where);
            require(meets(contact - slack, 1e-9),
                    "overlapping boxes: " + where);
            require(!meets(contact + slack, 0.),
                    "separated boxes at margin 0: " + where);
            require(meets(contact + slack, 2e-3),
                    "boxes within the margin: " + where);
            ++checked;
          }
    }
  }
  require(checked == 64 * 4 * 26, "exact box cases");
}

// ---------------------------------------------------------------------------
// 3. Hulls of rotated polytopes and balls against the exact distance
// ---------------------------------------------------------------------------
void test_hull_supports() {
  Rng rng(7771);
  long intersecting = 0, separated = 0, within_margin = 0, ball_cases = 0;
  for (int k = 0; k < 4000; ++k) {
    const double scale_a = std::pow(10., uniform(rng, -0.5, 0.5));
    const double scale_b = std::pow(10., uniform(rng, -0.5, 0.5));
    const Coord3D offset_a = random_point(rng, k % 3 == 0 ? 1e3 : 5.);

    const auto make = [&](const Coord3D &offset, const double scale) {
      if (rng() % 6 == 0)
        return ball_body(offset, scale * uniform(rng, 0.3, 1.));
      return polytope_body(POLY_KINDS[pick(rng, POLY_KINDS.size())], rng, scale,
                           offset);
    };
    const TestBody a = make(offset_a, scale_a);
    const double reach = a.ball.radius + scale_b * 1.5;
    const V direction =
        random_unit(rng) * static_cast<LD>(uniform(rng, 0.1, 2.2) * reach);
    const TestBody b = make(offset_a + to_c3(direction), scale_b);
    ball_cases += a.round || b.round;

    const double margin = k % 2 ? 0.15 * std::min(scale_a, scale_b) : 0.;
    const LD gap = body_gap(a, b);
    const double band =
        (a.round || b.round ? 1e-7 : 1e-9) * reference_size(a, b);
    const bool got = bodies_meet(a, b, margin);
    const std::string where = a.label + " / " + b.label + " case " +
                              std::to_string(k) + " gap " +
                              std::to_string(static_cast<double>(gap)) +
                              " margin " + std::to_string(margin);
    if (gap <= static_cast<LD>(margin - band)) {
      require(got, "3D GJK misses an intersection: " + where);
      ++intersecting;
      within_margin += 0 < gap;
    } else if (static_cast<LD>(margin + band) < gap) {
      require(!got, "3D GJK reports a gap as an intersection: " + where);
      ++separated;
    }
  }
  require(200 < intersecting && 200 < separated,
          "one-sided hull cases: " + std::to_string(intersecting) + " " +
              std::to_string(separated));
  require(30 < within_margin,
          "hull cases within the margin: " + std::to_string(within_margin));
  require(100 < ball_cases, "ball cases: " + std::to_string(ball_cases));
}

// ---------------------------------------------------------------------------
// 4. Certification near touching, by contact type
// ---------------------------------------------------------------------------

void push_unit(std::vector<V> &directions, const V &direction) {
  if (1e-18L < vnorm2(direction))
    directions.push_back(vunit(direction));
}

// Indices of the points within tol of the farthest along m
std::vector<std::size_t> extreme_set(const Cloud &points, const V &m,
                                     const LD tol) {
  LD best = -std::numeric_limits<LD>::infinity();
  for (const auto &p : points)
    best = std::fmax(best, vdot(p, m));
  std::vector<std::size_t> set;
  for (std::size_t i = 0; i < points.size(); ++i)
    if (best - vdot(points[i], m) <= tol)
      set.push_back(i);
  return set;
}

// 0 for a point, 1 for collinear points, 2 for coplanar ones
int set_rank(const Cloud &points, const std::vector<std::size_t> &set) {
  if (set.size() == 1)
    return 0;
  const V e = points[set[1]] - points[set[0]];
  for (const std::size_t i : set) {
    const V d = points[i] - points[set[0]];
    if (1e-18L * vnorm2(e) * vnorm2(d) < vnorm2(vcross(e, d)))
      return 2;
  }
  return 1;
}

FeatureTable feature_table(const Cloud &p, Rng &rng) {
  V centroid{};
  for (const auto &q : p)
    centroid = centroid + q;
  centroid = centroid * (1 / static_cast<LD>(p.size()));

  std::vector<V> directions;
  for (std::size_t i = 0; i < p.size(); ++i) {
    push_unit(directions, p[i] - centroid);
    for (std::size_t j = i + 1; j < p.size(); ++j) {
      push_unit(directions, (p[i] + p[j]) * 0.5L - centroid);
      for (int r = 0; r < 3; ++r)
        push_unit(directions, perpendicular_unit(vunit(p[j] - p[i]), rng));
      for (std::size_t l = j + 1; l < p.size(); ++l) {
        const V normal = vcross(p[j] - p[i], p[l] - p[i]);
        push_unit(directions, normal);
        push_unit(directions, normal * -1.L);
      }
    }
  }
  for (int r = 0; r < 20; ++r)
    directions.push_back(random_unit(rng));

  FeatureTable table;
  for (const auto &m : directions) {
    const auto set = extreme_set(p, m, 1e-14L);
    const int rank = set_rank(p, set);
    V e{};
    if (rank == 1) {
      if (set.size() != 2)
        continue;
      const V along = vunit(p[set[1]] - p[set[0]]);
      e = vunit(along - m * vdot(along, m));
    }
    auto &entries = table[static_cast<std::size_t>(rank)];
    if (entries.size() < 64)
      entries.push_back({m, e});
  }
  return table;
}

// Random point of the hull of the indexed points (positive random weights)
V random_feature_point(const Cloud &points, const std::vector<std::size_t> &set,
                       Rng &rng) {
  V sum{};
  LD total = 0;
  for (const std::size_t i : set) {
    const LD w = uniform(rng, 0.05, 1.);
    sum = sum + points[i] * w;
    total += w;
  }
  return sum * (1 / total);
}

// Two polytopes in contact along the unit direction n (from a to b): b
// touches a in the planes n . x = const, with the feature of rank_a of a and
// the one of rank_b of b meeting at random points of both
struct Contact {
  Cloud a;
  Cloud b_touching;
  V n;
  std::string what;
};

std::optional<Contact> build_contact(Rng &rng, const PolyKind kind_a,
                                     const PolyKind kind_b, const int rank_a,
                                     const int rank_b) {
  const Cloud local_a = local_polytope(kind_a, rng);
  const Cloud local_b = local_polytope(kind_b, rng);
  const FeatureTable table_a = feature_table(local_a, rng);
  const FeatureTable table_b = feature_table(local_b, rng);
  const auto &entries_a = table_a[static_cast<std::size_t>(rank_a)];
  const auto &entries_b = table_b[static_cast<std::size_t>(rank_b)];
  if (entries_a.empty() || entries_b.empty())
    return std::nullopt;
  const FeatureEntry fa = entries_a[pick(rng, entries_a.size())];
  const FeatureEntry fb = entries_b[pick(rng, entries_b.size())];

  const V n = random_unit(rng);
  const V t = perpendicular_unit(n, rng);
  const LD phi = uniform(rng, 0.3, 3.1);
  const V t_b = t * std::cos(phi) + vcross(n, t) * std::sin(phi);
  const V e_a = rank_a == 1 ? fa.e : perpendicular_unit(fa.m, rng);
  const V e_b = rank_b == 1 ? fb.e : perpendicular_unit(fb.m, rng);
  const Rows3 rotation_a = frame_rotation(fa.m, e_a, n, t);
  const Rows3 rotation_b = frame_rotation(fb.m, e_b, n * -1.L, t_b);

  const LD scale_a = std::pow(10., uniform(rng, -2., 2.));
  const LD scale_b = std::pow(10., uniform(rng, -2., 2.));
  Cloud raw_a, raw_b;
  for (const auto &p : local_a)
    raw_a.push_back(matrix_apply(rotation_a, p) * scale_a);
  for (const auto &p : local_b)
    raw_b.push_back(matrix_apply(rotation_b, p) * scale_b);

  // Features farthest along n (a) and along -n (b)
  const auto set_a = extreme_set(raw_a, n, 1e-14L * scale_a);
  const auto set_b = extreme_set(raw_b, n * -1.L, 1e-14L * scale_b);
  const V touch_a = random_feature_point(raw_a, set_a, rng);
  const V touch_b = random_feature_point(raw_b, set_b, rng);

  const V offset = to_v(random_point(rng, 1e3));
  Contact contact;
  contact.n = n;
  for (const auto &p : raw_a)
    contact.a.push_back(offset + p);
  for (const auto &p : raw_b)
    contact.b_touching.push_back(offset + p + (touch_a - touch_b));
  contact.what = kind_name(kind_a) + " (rank " + std::to_string(rank_a) +
                 ") / " + kind_name(kind_b) + " (rank " +
                 std::to_string(rank_b) + ")";
  return contact;
}

struct ContactType {
  const char *name;
  int rank_a;
  int rank_b;
  bool both_orders;
};

constexpr std::array<ContactType, 6> CONTACT_TYPES{
    {{"vertex-vertex", 0, 0, false},
     {"vertex-edge", 0, 1, true},
     {"edge-edge", 1, 1, false},
     {"vertex-face", 0, 2, true},
     {"edge-face", 1, 2, true},
     {"face-face", 2, 2, false}}};

struct GapProbe {
  double factor; // of delta
  bool intersects;
  const char *name;
};

constexpr std::array<GapProbe, 5> GAP_PROBES{{{4., false, "gap 4 delta"},
                                              {0., true, "touching"},
                                              {0.5, true, "gap delta / 2"},
                                              {-1., true, "overlap delta"},
                                              {-3., true, "overlap 3 delta"}}};

void test_contact_certification() {
  Rng rng(9090);
  constexpr long CASES_PER_TYPE = 250;
  std::array<long, CONTACT_TYPES.size()> counts{};

  for (std::size_t type = 0; type < CONTACT_TYPES.size(); ++type) {
    const ContactType &spec = CONTACT_TYPES[type];
    for (long tries = 0; counts[type] < CASES_PER_TYPE && tries < 40000;
         ++tries) {
      int rank_a = spec.rank_a, rank_b = spec.rank_b;
      if (spec.both_orders && rng() % 2 == 0)
        std::swap(rank_a, rank_b);
      const auto contact = build_contact(
          rng, POLY_KINDS[pick(rng, POLY_KINDS.size())],
          POLY_KINDS[pick(rng, POLY_KINDS.size())], rank_a, rank_b);
      if (!contact)
        continue;

      std::vector<Coord3D> points_a;
      for (const auto &p : contact->a)
        points_a.push_back(to_c3(p));
      const Ball ball_a = bounding_ball(points_a);
      const Hull hull_a{points_a};

      const auto points_b_at = [&](const LD gap) {
        std::vector<Coord3D> points;
        for (const auto &p : contact->b_touching)
          points.push_back(to_c3(p + contact->n * gap));
        return points;
      };
      const Ball ball_b = bounding_ball(points_b_at(0));
      const double delta = filter_margin(ball_a, ball_b);
      const double size = std::sqrt(
          ball_a.radius * ball_a.radius + vector_norm2(ball_a.center) +
          ball_b.radius * ball_b.radius + vector_norm2(ball_b.center));

      for (const auto &probe : GAP_PROBES) {
        const LD gap = static_cast<LD>(probe.factor) * delta;
        const Hull hull_b{points_b_at(gap)};

        // The construction itself: the gap along n is the one asked for
        const Extent ea = project(to_cloud(hull_a.points), contact->n);
        const Extent eb = project(to_cloud(hull_b.points), contact->n);
        require(std::fabs((eb.lo - ea.hi) - gap) <= 5e-13L * size,
                "contact construction is off: " + contact->what);

        const bool got = convex_sets_intersect(
            hull_a, hull_b, ball_a.center - ball_b.center, delta);
        // Pushing B past the contact along n makes the sets overlap only for
        // some contacts (not past a sharp vertex, nor for flat sets): the
        // overlap probes take their verdict from the separation oracle and
        // skip the band around the margin
        bool expected = probe.intersects;
        if (probe.factor < 0) {
          const LD exact =
              separation(to_cloud(hull_a.points), to_cloud(hull_b.points));
          const LD band = 1e-6L * static_cast<LD>(delta);
          if (std::fabs(exact - static_cast<LD>(delta)) <= band)
            continue;
          expected = exact < static_cast<LD>(delta);
        }
        require(got == expected,
                std::string("GJK ") +
                    (expected ? "misses an intersection"
                              : "does not certify the gap") +
                    " (" + probe.name + ", contact " + spec.name + ", case " +
                    std::to_string(counts[type]) + ", try " +
                    std::to_string(tries) + "): " + contact->what + ", delta " +
                    std::to_string(delta));
      }
      ++counts[type];
    }
    require(200 <= counts[type], std::string("too few contacts of type ") +
                                     spec.name + ": " +
                                     std::to_string(counts[type]));
  }
}

// ---------------------------------------------------------------------------
// 5. closest_point_on_simplex_to_origin with a tiny closest point
// ---------------------------------------------------------------------------

// Solves the n x n system (n <= 3) by Gaussian elimination with partial
// pivoting; false for an exactly singular one
bool solve_linear(const std::array<std::array<LD, 3>, 3> &matrix,
                  const std::array<LD, 3> &rhs_in, const std::size_t n,
                  std::array<LD, 3> &x) {
  // Working copies, reduced in place
  std::array<std::array<LD, 3>, 3> a = matrix;
  std::array<LD, 3> b = rhs_in;
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < n; ++r)
      if (std::fabs(a[pivot][col]) < std::fabs(a[r][col]))
        pivot = r;
    if (!(0 < std::fabs(a[pivot][col])))
      return false;
    std::swap(a[col], a[pivot]);
    std::swap(b[col], b[pivot]);
    for (std::size_t r = col + 1; r < n; ++r) {
      const LD f = a[r][col] / a[col][col];
      for (std::size_t c = col; c < n; ++c)
        a[r][c] -= f * a[col][c];
      b[r] -= f * b[col];
    }
  }
  for (std::size_t i = n; 0 < i--;) {
    LD sum = b[i];
    for (std::size_t c = i + 1; c < n; ++c)
      sum -= a[i][c] * x[c];
    x[i] = sum / a[i][i];
  }
  return true;
}

// Closest point of the hull of up to four points to the origin: the best of
// the least-squares points of every subset (normal equations in the edges
// from its first point) whose barycentric coordinates are not negative
V closest_in_hull(const Cloud &points) {
  V best{};
  LD best2 = std::numeric_limits<LD>::infinity();
  const std::size_t n = points.size();
  for (unsigned mask = 1; mask < (1U << n); ++mask) {
    std::vector<std::size_t> index;
    for (std::size_t i = 0; i < n; ++i)
      if (((mask >> i) & 1U) != 0)
        index.push_back(i);
    const std::size_t k = index.size();
    V candidate = points[index[0]];
    bool valid = true;
    if (1 < k) {
      std::array<V, 3> edges{};
      std::array<std::array<LD, 3>, 3> gram{};
      std::array<LD, 3> rhs{}, mu{};
      for (std::size_t i = 0; i + 1 < k; ++i)
        edges[i] = points[index[i + 1]] - points[index[0]];
      for (std::size_t i = 0; i + 1 < k; ++i) {
        for (std::size_t j = 0; j + 1 < k; ++j)
          gram[i][j] = vdot(edges[i], edges[j]);
        rhs[i] = -vdot(edges[i], points[index[0]]);
      }
      if (!solve_linear(gram, rhs, k - 1, mu))
        continue;
      LD sum = 0;
      for (std::size_t i = 0; i + 1 < k; ++i) {
        valid = valid && -1e-15L <= mu[i];
        sum += mu[i];
        candidate = candidate + edges[i] * mu[i];
      }
      valid = valid && -1e-15L <= 1 - sum;
    }
    if (valid && vnorm2(candidate) < best2) {
      best = candidate;
      best2 = vnorm2(candidate);
    }
  }
  return best;
}

bool same_point(const Coord3D &a, const Coord3D &b) {
  return a.x_ == b.x_ && a.y_ == b.y_ && a.z_ == b.z_;
}

// Three points around the origin of the plane (basis u, w) with the origin
// well inside their triangle
std::vector<V> triangle_around_origin(Rng &rng, const V &n, const V &u,
                                      const V &w, const LD scale) {
  for (;;) {
    std::vector<V> p;
    for (int i = 0; i < 3; ++i) {
      const LD angle = 2 * PI * i / 3 + uniform(rng, -0.6, 0.6);
      const LD radius = scale * uniform(rng, 0.5, 1.5);
      p.push_back(u * (radius * std::cos(angle)) +
                  w * (radius * std::sin(angle)));
    }
    const LD area = vdot(vcross(p[1] - p[0], p[2] - p[0]), n);
    const LD la = vdot(vcross(p[1], p[2]), n) / area;
    const LD lb = vdot(vcross(p[2], p[0]), n) / area;
    const LD lc = vdot(vcross(p[0], p[1]), n) / area;
    if (0.05L <= la && 0.05L <= lb && 0.05L <= lc)
      return p;
  }
}

// Simplices with 2 to 4 vertices of which the first rank + 1 span the feature
// holding the closest point d n, d / scale between 1e-8 and 1e-3, the others
// at least 0.2 scale beyond the plane n . x = d. The feature must be kept,
// the point orthogonal to it (the direction survives the cancellation) and
// its length the exact distance.
void test_simplex_near_touching() {
  Rng rng(5150);
  std::array<std::array<long, 3>, 5> combos{};
  for (int k = 0; k < 6000; ++k) {
    const std::size_t vertices = 2 + static_cast<std::size_t>(k % 3);
    const int rank = uniform_int(
        rng, 0, static_cast<int>(std::min<std::size_t>(vertices - 1, 2)));
    const LD scale = std::pow(10., uniform(rng, -1., 3.));
    const LD d = scale * std::pow(10., uniform(rng, -8., -3.));
    const V n = random_unit(rng);
    const V u = perpendicular_unit(n, rng);
    const V w = vcross(n, u);

    std::vector<V> in_plane;
    if (rank == 0)
      in_plane.push_back(V{});
    else if (rank == 1) {
      in_plane.push_back(u * (-scale * uniform(rng, 0.3, 1.2)));
      in_plane.push_back(u * (scale * uniform(rng, 0.3, 1.2)));
    } else
      in_plane = triangle_around_origin(rng, n, u, w, scale);

    struct Vertex {
      Coord3D point;
      bool in_feature;
    };
    std::vector<Vertex> simplex_vertices;
    for (const auto &p : in_plane)
      simplex_vertices.push_back({to_c3(n * d + p), true});
    while (simplex_vertices.size() < vertices) {
      const LD h = scale * uniform(rng, 0.2, 1.);
      simplex_vertices.push_back(
          {to_c3(n * (d + h) + u * (scale * uniform(rng, -1., 1.)) +
                 w * (scale * uniform(rng, -1., 1.))),
           false});
    }
    std::shuffle(simplex_vertices.begin(), simplex_vertices.end(), rng);

    Simplex<Coord3D> simplex;
    Cloud exact;
    std::vector<Coord3D> feature;
    for (const auto &vertex : simplex_vertices) {
      simplex.push(vertex.point);
      exact.push_back(to_v(vertex.point));
      if (vertex.in_feature)
        feature.push_back(vertex.point);
    }
    const std::string where = "simplex with " + std::to_string(vertices) +
                              " vertices, feature rank " +
                              std::to_string(rank) + ", case " +
                              std::to_string(k);

    const Coord3D v = closest_point_on_simplex_to_origin(simplex);
    require(finite(v), "non-finite closest point: " + where);

    const V closest = closest_in_hull(exact);
    const LD expected = vnorm(closest);
    require(std::fabs(vnorm(to_v(v)) - expected) <= 1e-6L * expected,
            "closest distance off: " + where);
    require(vnorm(to_v(v) - closest) <= 1e-5L * expected,
            "closest point off: " + where);

    require(static_cast<std::size_t>(simplex.size_) == feature.size(),
            "wrong feature size kept: " + where);
    for (dim_t i = 0; i < simplex.size_; ++i) {
      const bool known =
          std::any_of(feature.begin(), feature.end(), [&](const Coord3D &f) {
            return same_point(f, simplex.points_[i]);
          });
      require(known, "kept vertex is not of the feature: " + where);
    }
    if (simplex.size_ == 1)
      require(same_point(v, simplex.points_[0]), "vertex feature: " + where);
    for (dim_t i = 1; i < simplex.size_; ++i) {
      const V e = to_v(simplex.points_[i]) - to_v(simplex.points_[0]);
      require(std::fabs(vdot(to_v(v), e)) <= 1e-12L * vnorm(to_v(v)) * vnorm(e),
              "closest point not orthogonal to the feature: " + where);
    }
    ++combos[vertices][static_cast<std::size_t>(rank)];
  }
  for (std::size_t vertices = 2; vertices <= 4; ++vertices)
    for (std::size_t rank = 0; rank < std::min<std::size_t>(vertices, 3);
         ++rank)
      require(100 < combos[vertices][rank],
              "simplex combination never occurs: " + std::to_string(vertices) +
                  " vertices, rank " + std::to_string(rank));
}

// ---------------------------------------------------------------------------
// 6. Random rotated polytopes against the separating-axis oracle
// ---------------------------------------------------------------------------
void test_random_polytopes() {
  Rng rng(1234);
  long disjoint_checks = 0, overlap_checks = 0, margin_checks = 0;
  for (int k = 0; k < 3000; ++k) {
    const double scale_a = std::pow(10., uniform(rng, -0.5, 0.5));
    const double scale_b = std::pow(10., uniform(rng, -0.5, 0.5));
    const Coord3D offset_a = random_point(rng, k % 2 ? 1e3 : 3.);
    const TestBody a = polytope_body(POLY_KINDS[pick(rng, POLY_KINDS.size())],
                                     rng, scale_a, offset_a);
    const double reach = a.ball.radius + scale_b * 1.5;
    const Coord3D displacement = to_c3(
        random_unit(rng) * static_cast<LD>(uniform(rng, 0.2, 2.0) * reach));
    const TestBody b = polytope_body(POLY_KINDS[pick(rng, POLY_KINDS.size())],
                                     rng, scale_b, offset_a + displacement);
    const LD gap = separation(a.exact, b.exact);
    const double band = 1e-9 * reference_size(a, b);
    const std::string where = a.label + " / " + b.label + " case " +
                              std::to_string(k) + " gap " +
                              std::to_string(static_cast<double>(gap));

    // Margin zero: separated iff the oracle finds a separating axis
    if (static_cast<LD>(band) < gap) {
      require(!bodies_meet(a, b, 0.),
              "GJK reports a gap at margin 0: " + where);
      ++disjoint_checks;
    } else if (gap <= -static_cast<LD>(band)) {
      require(bodies_meet(a, b, 0.), "GJK misses an overlap: " + where);
      require(bodies_meet(a, b, 0.1 * scale_a),
              "GJK misses an overlap: " + where);
      ++overlap_checks;
    }

    // Margins just below and above the exact gap
    if (0 < gap)
      for (const double eps : {1e-3, 1e-5}) {
        const double g = static_cast<double>(gap);
        if (!(2 * band < g * eps))
          continue;
        require(!bodies_meet(a, b, g * (1 - eps)),
                "GJK does not certify a gap just above the margin: " + where);
        require(bodies_meet(a, b, g * (1 + eps)),
                "GJK certifies a gap within the margin: " + where);
        ++margin_checks;
      }
  }
  require(400 < disjoint_checks && 300 < overlap_checks && 400 < margin_checks,
          "one-sided polytope cases: " + std::to_string(disjoint_checks) + " " +
              std::to_string(overlap_checks) + " " +
              std::to_string(margin_checks));
}
} // namespace

int main() {
  test_rotation();
  test_exact_boxes();
  test_hull_supports();
  test_contact_certification();
  test_simplex_near_touching();
  test_random_polytopes();
  return 0;
}
