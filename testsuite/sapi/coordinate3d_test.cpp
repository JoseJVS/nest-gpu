/*
 *  coordinate3d_test.cpp
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

// The 3D primitives of coordinate_geometry.h, each against an oracle built
// with different math (plain arrays, explicit matrix products, Gaussian
// elimination, half-spaces, brute-force sampling, optimality certificates):
//   triple_product                  determinant, cyclic symmetry, coplanarity
//   create_angular_offset            exact multiples of 90 degrees
//   create_rotation_3D and the rotate_displacement pair
//                                   elementary matrix products, orthonormality,
//                                   round trips, the 2D rotation for alpha
//                                   only, exact signed permutations
//   algebraic_projection_comparison<false/true> (3D)
//                                   exact integer lattice of weights around the
//                                   parallelepiped and the tetrahedron, random
//                                   shapes against the inverse matrix
//   circumscribed_center (3 and 4 points)
//                                   equidistance and the linear systems
//   smallest_enclosing_ball         containment, diameter and Jung bounds,
//                                   optimality certificate, perturbations
//   convex_polygon_distance2, prism_distance2
//                                   half-planes, clamped edge projections and
//                                   boundary or surface sampling
//   projection_coord                orthogonality, collinearity, clamping

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include "coordinate_geometry.h"
#include "coordinates.h"
#include "numerics.h"

using namespace sapi;

namespace {
using Rng = std::mt19937_64;
using V2 = std::array<double, 2>;
using V3 = std::array<double, 3>;
using Mat3 = std::array<V3, 3>; // by rows
using IMat3 = std::array<std::array<int, 3>, 3>;

// Prints what failed and aborts; independent of NDEBUG
void require(const bool condition, const std::string &what) {
  if (!condition) {
    std::fprintf(stderr, "coordinate3d_test check failed: %s\n", what.c_str());
    std::abort();
  }
}

double uniform(Rng &rng, const double lo, const double hi) {
  return std::uniform_real_distribution<double>(lo, hi)(rng);
}

int random_int(Rng &rng, const int lo, const int hi) {
  return std::uniform_int_distribution<int>(lo, hi)(rng);
}

V3 random_v3(Rng &rng, const double lo, const double hi) {
  return {uniform(rng, lo, hi), uniform(rng, lo, hi), uniform(rng, lo, hi)};
}

bool close(const double x, const double y, const double tol) {
  return std::fabs(x - y) <= tol * (1. + std::fabs(x) + std::fabs(y));
}

Coord3D to_c3(const V3 &v) { return construct_coord_3D(v[0], v[1], v[2]); }

V3 from_c3(const Coord3D &c) { return {c.x_, c.y_, c.z_}; }

V3 add(const V3 &a, const V3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

V3 sub(const V3 &a, const V3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

V3 scale(const V3 &a, const double s) { return {a[0] * s, a[1] * s, a[2] * s}; }

double dot3(const V3 &a, const V3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double dist2_3(const V3 &a, const V3 &b) {
  const V3 d = sub(a, b);
  return dot3(d, d);
}

bool close3(const V3 &a, const V3 &b, const double tol) {
  return close(a[0], b[0], tol) && close(a[1], b[1], tol) &&
         close(a[2], b[2], tol);
}

double det3(const V3 &a, const V3 &b, const V3 &c) {
  return a[0] * (b[1] * c[2] - b[2] * c[1]) -
         a[1] * (b[0] * c[2] - b[2] * c[0]) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

double deg2rad(const double degrees) {
  return degrees * std::numbers::pi / 180.;
}

// Elementary rotations by explicit matrices
Mat3 rot_z(const double deg) {
  const double c = std::cos(deg2rad(deg)), s = std::sin(deg2rad(deg));
  return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
}

Mat3 rot_y(const double deg) {
  const double c = std::cos(deg2rad(deg)), s = std::sin(deg2rad(deg));
  return {{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}};
}

Mat3 rot_x(const double deg) {
  const double c = std::cos(deg2rad(deg)), s = std::sin(deg2rad(deg));
  return {{{1, 0, 0}, {0, c, -s}, {0, s, c}}};
}

Mat3 mat_mul(const Mat3 &a, const Mat3 &b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        r[i][j] += a[i][k] * b[k][j];
  return r;
}

Mat3 transpose(const Mat3 &a) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i][j] = a[j][i];
  return r;
}

V3 mat_vec(const Mat3 &a, const V3 &v) {
  return {dot3(a[0], v), dot3(a[1], v), dot3(a[2], v)};
}

Mat3 reference_rotation(const int alpha, const int beta, const int gamma) {
  return mat_mul(mat_mul(rot_z(alpha), rot_y(beta)), rot_x(gamma));
}

Mat3 to_mat(const Rotation3D &r) {
  return {from_c3(r[0]), from_c3(r[1]), from_c3(r[2])};
}

// Gaussian elimination with partial pivoting, false for a singular matrix
bool solve3(const Mat3 &matrix, const V3 &rhs, V3 &x) {
  // Working copies, reduced in place
  Mat3 a = matrix;
  V3 b = rhs;
  for (int col = 0; col < 3; ++col) {
    int pivot = col;
    for (int row = col + 1; row < 3; ++row)
      if (std::fabs(a[row][col]) > std::fabs(a[pivot][col]))
        pivot = row;
    if (std::fabs(a[pivot][col]) < 1e-14)
      return false;
    std::swap(a[col], a[pivot]);
    std::swap(b[col], b[pivot]);
    for (int row = col + 1; row < 3; ++row) {
      const double f = a[row][col] / a[col][col];
      for (int k = col; k < 3; ++k)
        a[row][k] -= f * a[col][k];
      b[row] -= f * b[col];
    }
  }
  for (int row = 2; row >= 0; --row) {
    double s = b[row];
    for (int k = row + 1; k < 3; ++k)
      s -= a[row][k] * x[k];
    x[row] = s / a[row][row];
  }
  return true;
}

// Integer angles, the multiples of 90 in [-360, 360]
std::vector<int> right_angles() {
  std::vector<int> v;
  for (int k = -4; k <= 4; ++k)
    v.push_back(90 * k);
  return v;
}

// ---------------------------------------------------------------------------
// triple_product
// ---------------------------------------------------------------------------
void test_triple_product() {
  Rng rng(101);
  require(
      triple_product(to_c3({1, 0, 0}), to_c3({0, 1, 0}), to_c3({0, 0, 1})) == 1,
      "triple product of the axes");

  for (int iter = 0; iter < 1000; ++iter) {
    const V3 a = random_v3(rng, -3, 3), b = random_v3(rng, -3, 3),
             c = random_v3(rng, -3, 3);
    const Coord3D ca = to_c3(a), cb = to_c3(b), cc = to_c3(c);
    const double tp = triple_product(ca, cb, cc);
    require(close(tp, det3(a, b, c), 1e-12),
            "triple product is the determinant");
    require(close(tp,
                  determinant3x3(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1],
                                 c[2]),
                  1e-12),
            "triple product is determinant3x3");
    require(close(triple_product(cb, cc, ca), tp, 1e-12) &&
                close(triple_product(cc, ca, cb), tp, 1e-12),
            "triple product cyclic symmetry");
    require(close(triple_product(cb, ca, cc), -tp, 1e-12) &&
                close(triple_product(ca, cc, cb), -tp, 1e-12) &&
                close(triple_product(cc, cb, ca), -tp, 1e-12),
            "triple product antisymmetry");

    // Coplanar real vectors
    const V3 e =
        add(scale(a, uniform(rng, -2, 2)), scale(b, uniform(rng, -2, 2)));
    require(std::fabs(triple_product(ca, cb, to_c3(e))) < 1e-12,
            "coplanar real vectors have a null triple product");
  }

  // Coplanar integer vectors: exactly zero in every order
  for (int iter = 0; iter < 500; ++iter) {
    V3 a{}, b{};
    for (int k = 0; k < 3; ++k) {
      a[k] = random_int(rng, -5, 5);
      b[k] = random_int(rng, -5, 5);
    }
    const V3 c =
        add(scale(a, random_int(rng, -4, 4)), scale(b, random_int(rng, -4, 4)));
    const Coord3D ca = to_c3(a), cb = to_c3(b), cc = to_c3(c);
    require(
        triple_product(ca, cb, cc) == 0 && triple_product(cb, cc, ca) == 0 &&
            triple_product(cc, ca, cb) == 0 && triple_product(cb, ca, cc) == 0,
        "coplanar integer vectors have an exactly null triple product");
    // Repeated vector
    require(triple_product(ca, ca, cb) == 0 &&
                triple_product(ca, cb, cb) == 0 &&
                triple_product(ca, cb, ca) == 0,
            "repeated vector has an exactly null triple product");
  }
}

// ---------------------------------------------------------------------------
// create_angular_offset
// ---------------------------------------------------------------------------
void test_create_angular_offset() {
  const std::array<double, 4> cos_table{1, 0, -1, 0};
  const std::array<double, 4> sin_table{0, 1, 0, -1};
  for (int deg = -720; deg <= 720; ++deg) {
    const Coord2D o = create_angular_offset(static_cast<angle_t>(deg));
    if (deg % 90 == 0) {
      const int k = ((deg / 90) % 4 + 4) % 4;
      require(o.x_ == cos_table[static_cast<std::size_t>(k)] &&
                  o.y_ == sin_table[static_cast<std::size_t>(k)],
              "exact (cos, sin) at " + std::to_string(deg));
    } else {
      const double rad = deg2rad(deg);
      // The argument reaches 4 pi: its rounding differs between deg2rad and
      // radians by a few ulps of the argument
      require(std::fabs(o.x_ - std::cos(rad)) < 1e-14 &&
                  std::fabs(o.y_ - std::sin(rad)) < 1e-14,
              "(cos, sin) close at " + std::to_string(deg));
    }
    require(std::fabs(o.x_ * o.x_ + o.y_ * o.y_ - 1) < 1e-15,
            "unit angular offset at " + std::to_string(deg));
  }
}

// ---------------------------------------------------------------------------
// Rotation3D, rotate_displacement, inverse_rotate_displacement
// ---------------------------------------------------------------------------
IMat3 int_rot_z(const int deg) {
  const std::array<int, 4> c{1, 0, -1, 0}, s{0, 1, 0, -1};
  const auto k = static_cast<std::size_t>(((deg / 90) % 4 + 4) % 4);
  return {{{c[k], -s[k], 0}, {s[k], c[k], 0}, {0, 0, 1}}};
}

IMat3 int_rot_y(const int deg) {
  const std::array<int, 4> c{1, 0, -1, 0}, s{0, 1, 0, -1};
  const auto k = static_cast<std::size_t>(((deg / 90) % 4 + 4) % 4);
  return {{{c[k], 0, s[k]}, {0, 1, 0}, {-s[k], 0, c[k]}}};
}

IMat3 int_rot_x(const int deg) {
  const std::array<int, 4> c{1, 0, -1, 0}, s{0, 1, 0, -1};
  const auto k = static_cast<std::size_t>(((deg / 90) % 4 + 4) % 4);
  return {{{1, 0, 0}, {0, c[k], -s[k]}, {0, s[k], c[k]}}};
}

IMat3 int_mul(const IMat3 &a, const IMat3 &b) {
  IMat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +=
            a[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] *
            b[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
  return r;
}

void test_rotation_general() {
  Rng rng(202);
  for (int iter = 0; iter < 500; ++iter) {
    const int alpha = random_int(rng, -360, 360);
    const int beta = random_int(rng, -360, 360);
    const int gamma = random_int(rng, -360, 360);
    const Rotation3D rot = create_rotation_3D(static_cast<angle_t>(alpha),
                                              static_cast<angle_t>(beta),
                                              static_cast<angle_t>(gamma));
    const Mat3 m = to_mat(rot);
    const Mat3 ref = reference_rotation(alpha, beta, gamma);

    // Composition order Rz(alpha) Ry(beta) Rx(gamma)
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        require(std::fabs(m[i][j] - ref[i][j]) < 5e-15,
                "rotation entry against the elementary product");

    // Orthonormality and det = +1
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        require(std::fabs(dot3(m[i], m[j]) - (i == j ? 1. : 0.)) < 5e-15,
                "rotation rows are orthonormal");
    require(std::fabs(det3(m[0], m[1], m[2]) - 1) < 5e-15,
            "rotation determinant is +1");
    // Columns too (R^T R = I)
    const Mat3 mt = transpose(m);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        require(std::fabs(dot3(mt[i], mt[j]) - (i == j ? 1. : 0.)) < 5e-15,
                "rotation columns are orthonormal");

    // rotate = R^T d, inverse = R d
    const V3 d = random_v3(rng, -5, 5);
    const V3 w = from_c3(rotate_displacement(to_c3(d), rot));
    require(close3(w, mat_vec(mt, d), 1e-14), "rotate_displacement is R^T d");
    const V3 u = from_c3(inverse_rotate_displacement(to_c3(d), rot));
    require(close3(u, mat_vec(m, d), 1e-14),
            "inverse_rotate_displacement is R d");
    require(
        close3(from_c3(inverse_rotate_displacement(to_c3(w), rot)), d, 1e-14),
        "inverse of rotate round trip");
    require(close3(from_c3(rotate_displacement(to_c3(u), rot)), d, 1e-14),
            "rotate of inverse round trip");
    require(close(dot3(w, w), dot3(d, d), 1e-14), "rotation keeps lengths");

    // Axes: R e_j is column j, R^T e_j is row j
    const Coord3D ex = to_c3({1, 0, 0}), ey = to_c3({0, 1, 0}),
                  ez = to_c3({0, 0, 1});
    const Coord3D cx = inverse_rotate_displacement(ex, rot);
    const Coord3D cy = inverse_rotate_displacement(ey, rot);
    const Coord3D cz = inverse_rotate_displacement(ez, rot);
    require(from_c3(cx) == V3{m[0][0], m[1][0], m[2][0]} &&
                from_c3(cy) == V3{m[0][1], m[1][1], m[2][1]} &&
                from_c3(cz) == V3{m[0][2], m[1][2], m[2][2]},
            "inverse rotation of the axes is the columns");
    require(from_c3(rotate_displacement(ex, rot)) == m[0] &&
                from_c3(rotate_displacement(ey, rot)) == m[1] &&
                from_c3(rotate_displacement(ez, rot)) == m[2],
            "rotation of the axes is the rows");
  }
}

void test_rotation_alpha_only() {
  Rng rng(203);
  for (int alpha = -360; alpha <= 360; alpha += 3) {
    const Rotation3D rot =
        create_rotation_3D(static_cast<angle_t>(alpha), 0, 0);
    const V3 d = random_v3(rng, -5, 5);
    const Coord3D r3 = rotate_displacement(to_c3(d), rot);
    const Coord2D r2 =
        rotate_displacement(construct_coord_2D(d[0], d[1]),
                            create_angular_offset(static_cast<space_t>(alpha)));
    require(close(r3.x_, r2.x_, 1e-14) && close(r3.y_, r2.y_, 1e-14),
            "alpha only rotation equals the 2D one at " +
                std::to_string(alpha));
    require(r3.z_ == d[2], "alpha only rotation keeps z exactly");
  }
}

void test_rotation_right_angles() {
  const std::vector<int> angles = right_angles();
  const V3 v{3, -5, 7};
  for (const int alpha : angles)
    for (const int beta : angles)
      for (const int gamma : angles) {
        const Rotation3D rot = create_rotation_3D(static_cast<angle_t>(alpha),
                                                  static_cast<angle_t>(beta),
                                                  static_cast<angle_t>(gamma));
        const IMat3 ref = int_mul(int_mul(int_rot_z(alpha), int_rot_y(beta)),
                                  int_rot_x(gamma));
        const Mat3 m = to_mat(rot);
        std::array<int, 3> row_nonzero{}, col_nonzero{};
        for (std::size_t i = 0; i < 3; ++i)
          for (std::size_t j = 0; j < 3; ++j) {
            require(m[i][j] == static_cast<double>(ref[i][j]),
                    "right angle rotation entry is exact");
            require(m[i][j] == 0 || m[i][j] == 1 || m[i][j] == -1,
                    "right angle rotation entry is 0 or +-1");
            row_nonzero[i] += m[i][j] != 0;
            col_nonzero[j] += m[i][j] != 0;
          }
        for (std::size_t i = 0; i < 3; ++i)
          require(row_nonzero[i] == 1 && col_nonzero[i] == 1,
                  "right angle rotation is a signed permutation");

        // Integer vectors stay integer, exactly the integer matrix product
        const V3 w = from_c3(rotate_displacement(to_c3(v), rot));
        const V3 u = from_c3(inverse_rotate_displacement(to_c3(v), rot));
        for (std::size_t i = 0; i < 3; ++i) {
          const double expected_w =
              ref[0][i] * v[0] + ref[1][i] * v[1] + ref[2][i] * v[2];
          const double expected_u =
              ref[i][0] * v[0] + ref[i][1] * v[1] + ref[i][2] * v[2];
          require(w[i] == expected_w && w[i] == std::round(w[i]),
                  "right angle rotation of an integer vector is exact");
          require(u[i] == expected_u && u[i] == std::round(u[i]),
                  "right angle inverse rotation of an integer vector is exact");
        }
      }
}

// ---------------------------------------------------------------------------
// algebraic_projection_comparison (3D)
// ---------------------------------------------------------------------------
// Weights in sixths of the edges, integer edges multiples of 6: every vector
// and triple product is an exact integer, so the closed boundary (vertices,
// edge midpoints, face centroids) and points just outside it are exact cases
void test_projection_comparison_lattice() {
  Rng rng(301);
  int tested_shapes = 0;
  int inside_par = 0, outside_par = 0, inside_tet = 0, outside_tet = 0;
  for (int iter = 0; iter < 300; ++iter) {
    V3 k[3];
    for (auto &row : k)
      row = {static_cast<double>(random_int(rng, -3, 3)),
             static_cast<double>(random_int(rng, -3, 3)),
             static_cast<double>(random_int(rng, -3, 3))};
    if (det3(k[0], k[1], k[2]) == 0)
      continue;
    ++tested_shapes;
    const Coord3D BC = to_c3(scale(k[0], 6)), BD = to_c3(scale(k[1], 6)),
                  BE = to_c3(scale(k[2], 6));
    const space_t det = triple_product(BC, BD, BE);
    require(det == 216 * det3(k[0], k[1], k[2]), "lattice determinant");

    for (int ns = -2; ns <= 8; ++ns)
      for (int nt = -2; nt <= 8; ++nt)
        for (int nu = -2; nu <= 8; ++nu) {
          const V3 ba =
              add(add(scale(k[0], ns), scale(k[1], nt)), scale(k[2], nu));
          const bool in_box =
              0 <= ns && ns <= 6 && 0 <= nt && nt <= 6 && 0 <= nu && nu <= 6;
          const bool in_tet = in_box && ns + nt + nu <= 6;
          const bool par = algebraic_projection_comparison<false>(to_c3(ba), BC,
                                                                  BD, BE, det);
          const bool tet =
              algebraic_projection_comparison<true>(to_c3(ba), BC, BD, BE, det);
          const std::string what = " at weights " + std::to_string(ns) + " " +
                                   std::to_string(nt) + " " +
                                   std::to_string(nu) + " (sixths)";
          require(par == in_box, "parallelepiped lattice point" + what);
          require(tet == in_tet, "tetrahedron lattice point" + what);
          inside_par += in_box;
          outside_par += !in_box;
          inside_tet += in_tet;
          outside_tet += !in_tet;
        }
  }
  require(100 < tested_shapes, "enough lattice shapes");
  require(0 < inside_par && 0 < outside_par && 0 < inside_tet &&
              0 < outside_tet,
          "lattice has points of both classes");
}

struct RandomCell {
  V3 corner;
  Mat3 edges; // BC, BD, BE as rows
  double det;
};

RandomCell random_cell(Rng &rng) {
  while (true) {
    RandomCell cell{};
    cell.corner = random_v3(rng, -2, 2);
    if (random_int(rng, 0, 1) == 0) {
      for (auto &edge : cell.edges)
        edge = random_v3(rng, -2, 2);
    } else {
      // Rotated box with independent edge lengths: columns of the rotation
      const Mat3 r = transpose(reference_rotation(random_int(rng, -360, 360),
                                                  random_int(rng, -360, 360),
                                                  random_int(rng, -360, 360)));
      for (std::size_t i = 0; i < 3; ++i)
        cell.edges[i] = scale(r[i], uniform(rng, 0.5, 2));
    }
    cell.det = det3(cell.edges[0], cell.edges[1], cell.edges[2]);
    if (0.3 < std::fabs(cell.det))
      return cell;
  }
}

bool far_from(const double x, const double target) {
  return 1e-7 < std::fabs(x - target);
}

void test_projection_comparison_random() {
  Rng rng(302);
  int inside_par = 0, outside_par = 0, inside_tet = 0, outside_tet = 0;
  for (int iter = 0; iter < 3000; ++iter) {
    const RandomCell cell = random_cell(rng);
    const Coord3D BC = to_c3(cell.edges[0]), BD = to_c3(cell.edges[1]),
                  BE = to_c3(cell.edges[2]);
    const space_t det = triple_product(BC, BD, BE);

    // A random point, classified by the inverse matrix (columns are the edges)
    const V3 w = random_v3(rng, -0.4, 1.4);
    const V3 a =
        add(cell.corner,
            add(add(scale(cell.edges[0], w[0]), scale(cell.edges[1], w[1])),
                scale(cell.edges[2], w[2])));
    const V3 ba = sub(a, cell.corner);
    V3 x{};
    require(solve3(transpose(cell.edges), ba, x),
            "edge matrix is not singular");
    bool clear = true;
    for (const double xi : x)
      clear &= far_from(xi, 0) && far_from(xi, 1);
    if (clear) {
      const bool in_box = 0 <= x[0] && x[0] <= 1 && 0 <= x[1] && x[1] <= 1 &&
                          0 <= x[2] && x[2] <= 1;
      require(algebraic_projection_comparison<false>(to_c3(ba), BC, BD, BE,
                                                     det) == in_box,
              "random point against the parallelepiped");
      inside_par += in_box;
      outside_par += !in_box;
    }
    const V3 wt = random_v3(rng, -0.3, 0.8);
    const V3 at =
        add(cell.corner,
            add(add(scale(cell.edges[0], wt[0]), scale(cell.edges[1], wt[1])),
                scale(cell.edges[2], wt[2])));
    const V3 bat = sub(at, cell.corner);
    V3 xt{};
    require(solve3(transpose(cell.edges), bat, xt),
            "edge matrix is not singular");
    clear = far_from(xt[0] + xt[1] + xt[2], 1);
    for (const double xi : xt)
      clear &= far_from(xi, 0);
    if (clear) {
      const bool in_tet =
          0 <= xt[0] && 0 <= xt[1] && 0 <= xt[2] && xt[0] + xt[1] + xt[2] <= 1;
      require(algebraic_projection_comparison<true>(to_c3(bat), BC, BD, BE,
                                                    det) == in_tet,
              "random point against the tetrahedron");
      inside_tet += in_tet;
      outside_tet += !in_tet;
    }

    // Clear margins from every face: pushed out along each coefficient, and
    // pulled inside
    const double m = 1e-4;
    const auto point_at = [&](const V3 &coeffs) {
      return to_c3(add(
          add(scale(cell.edges[0], coeffs[0]), scale(cell.edges[1], coeffs[1])),
          scale(cell.edges[2], coeffs[2])));
    };
    const V3 inner{uniform(rng, m, 0.3), uniform(rng, m, 0.3),
                   uniform(rng, m, 0.3)};
    require(algebraic_projection_comparison<false>(point_at(inner), BC, BD, BE,
                                                   det) &&
                algebraic_projection_comparison<true>(point_at(inner), BC, BD,
                                                      BE, det),
            "interior point is inside");
    for (std::size_t axis = 0; axis < 3; ++axis) {
      V3 low = inner, high = inner;
      low[axis] = -m;
      high[axis] = 1 + m;
      require(!algebraic_projection_comparison<false>(point_at(low), BC, BD, BE,
                                                      det) &&
                  !algebraic_projection_comparison<false>(point_at(high), BC,
                                                          BD, BE, det) &&
                  !algebraic_projection_comparison<true>(point_at(low), BC, BD,
                                                         BE, det) &&
                  !algebraic_projection_comparison<true>(point_at(high), BC, BD,
                                                         BE, det),
              "point pushed out of a face is outside");
    }
    // Inside the box but past the slanted tetrahedron face
    const V3 beyond{0.4, 0.4, 0.4 + m};
    require(algebraic_projection_comparison<false>(point_at(beyond), BC, BD, BE,
                                                   det) &&
                !algebraic_projection_comparison<true>(point_at(beyond), BC, BD,
                                                       BE, det),
            "point beyond the slanted face is in the box only");
  }
  require(0 < inside_par && 0 < outside_par && 0 < inside_tet &&
              0 < outside_tet,
          "random points of both classes");
}

// ---------------------------------------------------------------------------
// Circumcenters
// ---------------------------------------------------------------------------
void test_circumcenters() {
  Rng rng(401);

  // 2D triangle: equidistance and the 2x2 linear system
  for (int iter = 0, done = 0; done < 500 && iter < 5000; ++iter) {
    const V3 a = random_v3(rng, -2, 2), b = random_v3(rng, -2, 2),
             c = random_v3(rng, -2, 2);
    const double area2 =
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if (std::fabs(area2) < 0.3)
      continue;
    ++done;
    const Coord2D o = circumscribed_center(construct_coord_2D(a[0], a[1]),
                                           construct_coord_2D(b[0], b[1]),
                                           construct_coord_2D(c[0], c[1]));
    const auto d2 = [&](const V3 &p) {
      return (o.x_ - p[0]) * (o.x_ - p[0]) + (o.y_ - p[1]) * (o.y_ - p[1]);
    };
    require(close(d2(a), d2(b), 1e-10) && close(d2(a), d2(c), 1e-10),
            "2D circumcenter is equidistant");
    // 2 (b - a) . x = |b|^2 - |a|^2, same for c, by Cramer
    const double r1 = b[0] * b[0] + b[1] * b[1] - a[0] * a[0] - a[1] * a[1];
    const double r2 = c[0] * c[0] + c[1] * c[1] - a[0] * a[0] - a[1] * a[1];
    const double m11 = 2 * (b[0] - a[0]), m12 = 2 * (b[1] - a[1]);
    const double m21 = 2 * (c[0] - a[0]), m22 = 2 * (c[1] - a[1]);
    const double det = m11 * m22 - m12 * m21;
    require(close(o.x_, (r1 * m22 - m12 * r2) / det, 1e-9) &&
                close(o.y_, (m11 * r2 - r1 * m21) / det, 1e-9),
            "2D circumcenter solves the linear system");
  }

  // 3D triangle: equidistance and in the plane
  for (int iter = 0, done = 0; done < 500 && iter < 5000; ++iter) {
    const V3 a = random_v3(rng, -2, 2), b = random_v3(rng, -2, 2),
             c = random_v3(rng, -2, 2);
    const V3 ab = sub(b, a), ac = sub(c, a);
    const V3 normal{ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0]};
    if (std::sqrt(dot3(normal, normal)) < 0.3)
      continue;
    ++done;
    const V3 o = from_c3(circumscribed_center(to_c3(a), to_c3(b), to_c3(c)));
    require(close(dist2_3(o, a), dist2_3(o, b), 1e-10) &&
                close(dist2_3(o, a), dist2_3(o, c), 1e-10),
            "3D triangle circumcenter is equidistant");
    require(std::fabs(dot3(sub(o, a), normal)) < 1e-9,
            "3D triangle circumcenter is in the plane");
  }

  // Tetrahedron: equidistance and the 3x3 linear system
  for (int iter = 0, done = 0; done < 500 && iter < 5000; ++iter) {
    const V3 a = random_v3(rng, -2, 2), b = random_v3(rng, -2, 2),
             c = random_v3(rng, -2, 2), d = random_v3(rng, -2, 2);
    if (std::fabs(det3(sub(b, a), sub(c, a), sub(d, a))) < 0.3)
      continue;
    ++done;
    const V3 o =
        from_c3(circumscribed_center(to_c3(a), to_c3(b), to_c3(c), to_c3(d)));
    require(close(dist2_3(o, a), dist2_3(o, b), 1e-9) &&
                close(dist2_3(o, a), dist2_3(o, c), 1e-9) &&
                close(dist2_3(o, a), dist2_3(o, d), 1e-9),
            "tetrahedron circumcenter is equidistant");
    Mat3 m{};
    V3 rhs{};
    const std::array<V3, 3> others{b, c, d};
    for (std::size_t j = 0; j < 3; ++j) {
      m[j] = scale(sub(others[j], a), 2);
      rhs[j] = dot3(others[j], others[j]) - dot3(a, a);
    }
    V3 x{};
    require(solve3(m, rhs, x), "circumcenter system is not singular");
    require(close3(o, x, 1e-8), "tetrahedron circumcenter solves the system");
  }
}

// ---------------------------------------------------------------------------
// smallest_enclosing_ball
// ---------------------------------------------------------------------------
template <Coordinate CoordT> CoordT make_coord(const V3 &p) {
  if constexpr (CoordT::D == 2)
    return construct_coord_2D(p[0], p[1]);
  else
    return construct_coord_3D(p[0], p[1], p[2]);
}

template <Coordinate CoordT> V3 to_v3(const CoordT &c) {
  if constexpr (CoordT::D == 2)
    return {c.x_, c.y_, 0.};
  else
    return {c.x_, c.y_, c.z_};
}

// Optimality certificate and bounds of a returned ball
template <Coordinate CoordT>
void check_ball(Rng &rng, const std::vector<V3> &points,
                const std::string &label) {
  constexpr std::size_t D = CoordT::D;
  std::vector<CoordT> coords;
  for (const auto &p : points)
    coords.push_back(make_coord<CoordT>(p));
  const CircumscribedRadius<CoordT> ball =
      smallest_enclosing_ball<CoordT>(coords);
  const V3 c = to_v3(ball.origin_);
  const double r2 = ball.radius2_;
  require(std::isfinite(r2) && std::isfinite(c[0]) && std::isfinite(c[1]) &&
              std::isfinite(c[2]),
          "finite ball, " + label);

  // Containment
  double diameter2 = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    require(distance2(ball.origin_, coords[i]) <= r2,
            "point inside the ball exactly, " + label);
    require(dist2_3(c, points[i]) <= r2 * (1 + 1e-12),
            "point inside the ball, " + label);
    for (std::size_t j = i + 1; j < points.size(); ++j)
      diameter2 = std::max(diameter2, dist2_3(points[i], points[j]));
  }

  // The diameter bounds the radius from below and Jung's theorem from above
  const double jung =
      static_cast<double>(D) / (2. * static_cast<double>(D + 1));
  require(r2 >= diameter2 / 4 * (1 - 1e-12),
          "radius below half diameter, " + label);
  require(r2 <= diameter2 * jung * (1 + 1e-9) + 1e-300,
          "radius above Jung's bound, " + label);

  if (r2 <= 0 || diameter2 == 0) {
    require(diameter2 == 0 && r2 == 0,
            "single location has a null ball, " + label);
    return;
  }
  const double r = std::sqrt(r2);

  // Optimality certificate: moving the center along d lowers the distance to
  // an active point (on the sphere) iff (c - p) . d < 0, so no d may have all
  // of them negative: the center is in the hull of the active points
  std::vector<V3> active;
  for (const auto &p : points)
    if (dist2_3(c, p) >= r2 * (1 - 1e-6))
      active.push_back(p);
  require(!active.empty(), "an active point exists, " + label);
  std::normal_distribution<double> gauss;
  for (int s = 0; s < 2000; ++s) {
    V3 dir{gauss(rng), gauss(rng), D == 3 ? gauss(rng) : 0.};
    const double n = std::sqrt(dot3(dir, dir));
    dir = scale(dir, 1 / n);
    double best = -std::numeric_limits<double>::infinity();
    for (const auto &p : active)
      best = std::max(best, dot3(sub(c, p), dir));
    require(best >= -1e-6 * r, "no descent direction, " + label);
  }

  // Perturbed centers do not lower the largest distance
  for (int s = 0; s < 300; ++s) {
    V3 dir{gauss(rng), gauss(rng), D == 3 ? gauss(rng) : 0.};
    dir = scale(dir, 1e-3 * r / std::sqrt(dot3(dir, dir)));
    const V3 moved = add(c, dir);
    double worst = 0;
    for (const auto &p : points)
      worst = std::max(worst, dist2_3(moved, p));
    require(worst >= r2 * (1 - 1e-9),
            "perturbed center is not better, " + label);
  }
}

// Families of point sets
std::vector<V3> make_points(Rng &rng, const std::size_t dim,
                            const std::size_t n, const int family) {
  const auto embed = [&](V3 p) {
    if (dim == 2)
      p[2] = 0;
    return p;
  };
  std::vector<V3> pts;
  const Mat3 frame = transpose(reference_rotation(
      random_int(rng, -360, 360), dim == 2 ? 0 : random_int(rng, -360, 360),
      dim == 2 ? 0 : random_int(rng, -360, 360)));
  const V3 origin = embed(random_v3(rng, -2, 2));
  const double size = uniform(rng, 0.1, 3);
  switch (family) {
  case 1: // Duplicates
    for (std::size_t i = 0; i < n; ++i)
      pts.push_back(embed(random_v3(rng, -1, 1)));
    for (std::size_t i = 1; i < n; ++i)
      if (random_int(rng, 0, 1) == 0)
        pts[i] = pts[static_cast<std::size_t>(
            random_int(rng, 0, static_cast<int>(i) - 1))];
    break;
  case 2: // All the same
    pts.assign(n, embed(random_v3(rng, -1, 1)));
    break;
  case 3:   // Collinear
  case 4: { // Nearly collinear
    const V3 dir = frame[0];
    const double noise =
        family == 3 ? 0. : (random_int(rng, 0, 1) ? 1e-9 : 1e-12);
    for (std::size_t i = 0; i < n; ++i) {
      V3 p = add(origin, scale(dir, size * uniform(rng, -1, 1)));
      for (auto &x : p)
        x += uniform(rng, -noise, noise);
      pts.push_back(embed(p));
    }
    break;
  }
  case 5: { // Coplanar in 3D (the plane of the frame), or planar in 2D
    for (std::size_t i = 0; i < n; ++i)
      pts.push_back(
          embed(add(origin, add(scale(frame[0], size * uniform(rng, -1, 1)),
                                scale(frame[1], size * uniform(rng, -1, 1))))));
    break;
  }
  case 6: { // Obtuse triangle, with an interior point when n = 4
    const double h = uniform(rng, 0.02, 0.45);
    const std::array<V3, 3> base{{{0, 0, 0}, {1, 0, 0}, {0.5, h, 0}}};
    for (std::size_t i = 0; i < 3 && i < n; ++i)
      pts.push_back(embed(add(origin, scale(mat_vec(frame, base[i]), size))));
    if (n == 4) {
      const double u = uniform(rng, 0, 1), v = uniform(rng, 0, 1 - u);
      pts.push_back(add(add(scale(pts[0], 1 - u - v), scale(pts[1], u)),
                        scale(pts[2], v)));
    }
    while (pts.size() < n)
      pts.push_back(pts.front());
    break;
  }
  default: // Uniform
    for (std::size_t i = 0; i < n; ++i)
      pts.push_back(embed(random_v3(rng, -1, 1)));
    break;
  }
  return pts;
}

void test_smallest_enclosing_ball_known() {
  // Single point, pair, equilateral triangle, right triangle, obtuse triangle,
  // regular tetrahedron, square, collinear triple, duplicates
  {
    const std::vector<Coord3D> p{construct_coord_3D(1, 2, 3)};
    const auto b = smallest_enclosing_ball<Coord3D>(p);
    require(b.radius2_ == 0 && b.origin_ == p[0], "ball of one point");
  }
  {
    const std::vector<Coord3D> p{construct_coord_3D(0, 0, 0),
                                 construct_coord_3D(2, 0, 0)};
    const auto b = smallest_enclosing_ball<Coord3D>(p);
    require(close(b.radius2_, 1, 1e-14) &&
                close3(to_v3(b.origin_), {1, 0, 0}, 1e-14),
            "ball of two points");
  }
  {
    const std::vector<Coord2D> p{construct_coord_2D(0, 0),
                                 construct_coord_2D(4, 0),
                                 construct_coord_2D(1, 0.5)};
    const auto b = smallest_enclosing_ball<Coord2D>(p);
    require(close(b.radius2_, 4, 1e-12) &&
                close3(to_v3(b.origin_), {2, 0, 0}, 1e-12),
            "ball of an obtuse triangle is the one of its longest edge");
  }
  {
    const std::vector<Coord2D> p{construct_coord_2D(0, 0),
                                 construct_coord_2D(2, 0),
                                 construct_coord_2D(0, 2)};
    const auto b = smallest_enclosing_ball<Coord2D>(p);
    require(close(b.radius2_, 2, 1e-12) &&
                close3(to_v3(b.origin_), {1, 1, 0}, 1e-12),
            "ball of a right triangle");
  }
  {
    const double h = std::sqrt(3.) / 2;
    const std::vector<Coord2D> p{construct_coord_2D(0, 0),
                                 construct_coord_2D(1, 0),
                                 construct_coord_2D(0.5, h)};
    const auto b = smallest_enclosing_ball<Coord2D>(p);
    require(close(b.radius2_, 1. / 3, 1e-12) &&
                close3(to_v3(b.origin_), {0.5, h / 3, 0}, 1e-12),
            "ball of an equilateral triangle");
  }
  {
    // Regular tetrahedron on alternate cube corners: center the cube center
    const std::vector<Coord3D> p{
        construct_coord_3D(1, 1, 1), construct_coord_3D(1, -1, -1),
        construct_coord_3D(-1, 1, -1), construct_coord_3D(-1, -1, 1)};
    const auto b = smallest_enclosing_ball<Coord3D>(p);
    require(close(b.radius2_, 3, 1e-12) &&
                close3(to_v3(b.origin_), {0, 0, 0}, 1e-12),
            "ball of a regular tetrahedron");
  }
  {
    // Coplanar square (4 points, flat): center of the square
    const std::vector<Coord3D> p{
        construct_coord_3D(0, 0, 5), construct_coord_3D(2, 0, 5),
        construct_coord_3D(2, 2, 5), construct_coord_3D(0, 2, 5)};
    const auto b = smallest_enclosing_ball<Coord3D>(p);
    require(close(b.radius2_, 2, 1e-12) &&
                close3(to_v3(b.origin_), {1, 1, 5}, 1e-12),
            "ball of a flat square");
  }
  {
    // Collinear triple, the middle one anywhere
    const std::vector<Coord3D> p{construct_coord_3D(0, 0, 0),
                                 construct_coord_3D(1, 1, 1),
                                 construct_coord_3D(4, 4, 4)};
    const auto b = smallest_enclosing_ball<Coord3D>(p);
    require(close(b.radius2_, 12, 1e-12) &&
                close3(to_v3(b.origin_), {2, 2, 2}, 1e-12),
            "ball of three collinear points");
  }
  {
    // Four points where one is interior to the ball of the other three
    const std::vector<Coord3D> p{
        construct_coord_3D(-1, 0, 0), construct_coord_3D(1, 0, 0),
        construct_coord_3D(0, 1, 0), construct_coord_3D(0.1, 0.1, 0.2)};
    const auto b = smallest_enclosing_ball<Coord3D>(p);
    require(close(b.radius2_, 1, 1e-12) &&
                close3(to_v3(b.origin_), {0, 0, 0}, 1e-12),
            "ball with an interior point");
  }
}

void test_smallest_enclosing_ball_random() {
  Rng rng(501);
  for (int family = 0; family <= 6; ++family) {
    for (std::size_t n = 1; n <= 3; ++n)
      for (int iter = 0; iter < 30; ++iter)
        check_ball<Coord2D>(rng, make_points(rng, 2, n, family),
                            "2D, " + std::to_string(n) + " points, family " +
                                std::to_string(family));
    for (std::size_t n = 1; n <= 4; ++n)
      for (int iter = 0; iter < 30; ++iter)
        check_ball<Coord3D>(rng, make_points(rng, 3, n, family),
                            "3D, " + std::to_string(n) + " points, family " +
                                std::to_string(family));
  }
}

// ---------------------------------------------------------------------------
// convex_polygon_distance2, prism_distance2
// ---------------------------------------------------------------------------

double cross2(const V2 &a, const V2 &b) { return a[0] * b[1] - a[1] * b[0]; }

V2 sub2(const V2 &a, const V2 &b) { return {a[0] - b[0], a[1] - b[1]}; }

double dist2_2(const V2 &a, const V2 &b) {
  const V2 d = sub2(a, b);
  return d[0] * d[0] + d[1] * d[1];
}

// Counterclockwise base shapes around the origin
std::vector<std::vector<V2>> base_polygons() {
  std::vector<std::vector<V2>> shapes;
  shapes.push_back({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}});
  shapes.push_back({{-1.5, -0.5}, {1.5, -0.5}, {1.5, 0.5}, {-1.5, 0.5}});
  shapes.push_back({{-1, -0.5}, {1.5, -0.3}, {0.2, 1.4}});
  std::vector<V2> hexagon;
  for (int k = 0; k < 6; ++k)
    hexagon.push_back(
        {1.2 * std::cos(deg2rad(60. * k)), 1.2 * std::sin(deg2rad(60. * k))});
  shapes.push_back(hexagon);
  return shapes;
}

std::vector<V2> transform_polygon(Rng &rng, const std::vector<V2> &base) {
  const double deg = random_int(rng, -360, 360);
  const double c = std::cos(deg2rad(deg)), s = std::sin(deg2rad(deg));
  const V2 shift{uniform(rng, -2, 2), uniform(rng, -2, 2)};
  std::vector<V2> poly;
  for (const auto &p : base)
    poly.push_back(
        {c * p[0] - s * p[1] + shift[0], s * p[0] + c * p[1] + shift[1]});
  // Cyclic shift of the first vertex
  std::rotate(poly.begin(),
              poly.begin() +
                  random_int(rng, 0, static_cast<int>(poly.size()) - 1),
              poly.end());
  return poly;
}

std::vector<Coord2D> to_coords2(const std::vector<V2> &poly) {
  std::vector<Coord2D> c;
  for (const auto &p : poly)
    c.push_back(construct_coord_2D(p[0], p[1]));
  return c;
}

bool inside_polygon(const V2 &p, const std::vector<V2> &poly) {
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const std::size_t j = (i + 1) % poly.size();
    if (cross2(sub2(poly[j], poly[i]), sub2(p, poly[i])) < 0)
      return false;
  }
  return true;
}

// Distance by clamped projection onto every edge, 0 inside
double polygon_distance2_oracle(const V2 &p, const std::vector<V2> &poly) {
  if (inside_polygon(p, poly))
    return 0;
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const std::size_t j = (i + 1) % poly.size();
    const V2 ab = sub2(poly[j], poly[i]);
    const double t =
        std::clamp(((p[0] - poly[i][0]) * ab[0] + (p[1] - poly[i][1]) * ab[1]) /
                       (ab[0] * ab[0] + ab[1] * ab[1]),
                   0., 1.);
    best = std::min(
        best, dist2_2(p, {poly[i][0] + t * ab[0], poly[i][1] + t * ab[1]}));
  }
  return best;
}

// Sampled boundary distance, an upper bound of the true one outside
double polygon_distance2_sampled(const V2 &p, const std::vector<V2> &poly,
                                 const int samples) {
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < poly.size(); ++i) {
    const std::size_t j = (i + 1) % poly.size();
    for (int k = 0; k <= samples; ++k) {
      const double t = static_cast<double>(k) / samples;
      best = std::min(best,
                      dist2_2(p, {poly[i][0] + t * (poly[j][0] - poly[i][0]),
                                  poly[i][1] + t * (poly[j][1] - poly[i][1])}));
    }
  }
  return best;
}

void test_convex_polygon_distance2() {
  Rng rng(601);
  const int samples = 200;
  for (const auto &base : base_polygons())
    for (int variant = 0; variant < 20; ++variant) {
      const std::vector<V2> poly = transform_polygon(rng, base);
      const std::vector<Coord2D> cpoly = to_coords2(poly);
      double longest = 0;
      for (std::size_t i = 0; i < poly.size(); ++i)
        longest = std::max(
            longest, std::sqrt(dist2_2(poly[i], poly[(i + 1) % poly.size()])));
      const double spacing = longest / samples;

      for (int s = 0; s < 60; ++s) {
        const V2 p{uniform(rng, -5, 5), uniform(rng, -5, 5)};
        const double d2 =
            convex_polygon_distance2(construct_coord_2D(p[0], p[1]), cpoly);
        const double expected = polygon_distance2_oracle(p, poly);
        require(std::fabs(d2 - expected) <= 1e-12 * (1 + expected),
                "polygon distance against the clamped projections");
        if (0 < expected) {
          const double sampled = polygon_distance2_sampled(p, poly, samples);
          require(d2 <= sampled + 1e-12, "polygon distance below sampled");
          require(std::sqrt(d2) >= std::sqrt(sampled) - spacing,
                  "polygon distance above the sampling bound");
        }
      }

      // Boundary, vertices and the centroid
      V2 centroid{0, 0};
      for (std::size_t i = 0; i < poly.size(); ++i) {
        const std::size_t j = (i + 1) % poly.size();
        centroid = {centroid[0] + poly[i][0] / static_cast<double>(poly.size()),
                    centroid[1] +
                        poly[i][1] / static_cast<double>(poly.size())};
        require(convex_polygon_distance2(
                    construct_coord_2D(poly[i][0], poly[i][1]), cpoly) < 1e-15,
                "vertex at distance 0");
        require(convex_polygon_distance2(
                    construct_coord_2D((poly[i][0] + poly[j][0]) / 2,
                                       (poly[i][1] + poly[j][1]) / 2),
                    cpoly) < 1e-15,
                "edge midpoint at distance 0");
      }
      require(convex_polygon_distance2(
                  construct_coord_2D(centroid[0], centroid[1]), cpoly) == 0,
              "centroid inside");
    }
}

struct PrismCase {
  std::vector<V2> face;
  double bottom, top;
  V3 origin;
  int alpha, beta, gamma;
};

PrismCase random_prism(Rng &rng, const std::vector<V2> &base) {
  PrismCase p;
  p.face = base;
  const V2 shift{uniform(rng, -0.5, 0.5), uniform(rng, -0.5, 0.5)};
  for (auto &v : p.face)
    v = {v[0] + shift[0], v[1] + shift[1]};
  p.bottom = uniform(rng, -1, 0.2);
  p.top = p.bottom + uniform(rng, 0.3, 2);
  p.origin = random_v3(rng, -2, 2);
  p.alpha = random_int(rng, -360, 360);
  p.beta = random_int(rng, -360, 360);
  p.gamma = random_int(rng, -360, 360);
  return p;
}

// Brute-force surface samples of the prism in its frame
std::vector<V3> prism_surface_samples(const PrismCase &p, const int n) {
  std::vector<V3> pts;
  for (std::size_t i = 0; i < p.face.size(); ++i) {
    const std::size_t j = (i + 1) % p.face.size();
    for (int u = 0; u <= n; ++u)
      for (int v = 0; v <= n; ++v) {
        const double t = static_cast<double>(u) / n;
        pts.push_back({p.face[i][0] + t * (p.face[j][0] - p.face[i][0]),
                       p.face[i][1] + t * (p.face[j][1] - p.face[i][1]),
                       p.bottom + (p.top - p.bottom) * v / n});
      }
  }
  double lo_x = 1e300, hi_x = -1e300, lo_y = 1e300, hi_y = -1e300;
  for (const auto &v : p.face) {
    lo_x = std::min(lo_x, v[0]);
    hi_x = std::max(hi_x, v[0]);
    lo_y = std::min(lo_y, v[1]);
    hi_y = std::max(hi_y, v[1]);
  }
  for (int u = 0; u <= n; ++u)
    for (int v = 0; v <= n; ++v) {
      const V2 q{lo_x + (hi_x - lo_x) * u / n, lo_y + (hi_y - lo_y) * v / n};
      if (inside_polygon(q, p.face)) {
        pts.push_back({q[0], q[1], p.bottom});
        pts.push_back({q[0], q[1], p.top});
      }
    }
  return pts;
}

void test_prism_distance2() {
  Rng rng(602);

  // Unit square face: the face distance plus the axial gap
  const std::array<Coord2D, 4> square{
      construct_coord_2D(0, 0), construct_coord_2D(1, 0),
      construct_coord_2D(1, 1), construct_coord_2D(0, 1)};
  require(
      close(prism_distance2(construct_coord_3D(0.5, 0.5, 0.5), square, 0., 1.),
            0, 1e-15),
      "inside");
  require(
      close(prism_distance2(construct_coord_3D(0.5, 0.5, -0.5), square, 0., 1.),
            0.25, 1e-15),
      "axial below");
  require(
      close(prism_distance2(construct_coord_3D(0.5, 0.5, 3.), square, 0., 1.),
            4, 1e-15),
      "axial above");
  require(
      close(prism_distance2(construct_coord_3D(0.5, 0.5, 1.), square, 0., 1.),
            0, 1e-15),
      "on the top");
  require(
      close(prism_distance2(construct_coord_3D(2., 0.5, 3.), square, 0., 1.), 5,
            1e-15),
      "face and axial gaps");

  int outside_cases = 0, inside_cases = 0;
  for (const auto &base : base_polygons())
    for (int variant = 0; variant < 20; ++variant) {
      const PrismCase p = random_prism(rng, base);
      const std::vector<Coord2D> face = to_coords2(p.face);
      const Rotation3D rot = create_rotation_3D(static_cast<angle_t>(p.alpha),
                                                static_cast<angle_t>(p.beta),
                                                static_cast<angle_t>(p.gamma));
      const Mat3 R = reference_rotation(p.alpha, p.beta, p.gamma);
      const std::vector<V3> samples = prism_surface_samples(p, 25);

      for (int s = 0; s < 8; ++s) {
        const V3 q = add(p.origin, random_v3(rng, -2.5, 2.5));
        const V3 displacement = sub(q, p.origin);

        // Frame mapping: the library against the explicit R^T d
        const Coord3D local = rotate_displacement(to_c3(displacement), rot);
        const V3 local_ref = mat_vec(transpose(R), displacement);
        require(close3(from_c3(local), local_ref, 1e-14),
                "prism frame mapping");

        const double d2 = prism_distance2(local, face, p.bottom, p.top);
        const double axial_gap =
            std::max({p.bottom - local_ref[2], local_ref[2] - p.top, 0.});
        const double expected =
            polygon_distance2_oracle({local_ref[0], local_ref[1]}, p.face) +
            axial_gap * axial_gap;
        require(std::fabs(d2 - expected) <= 1e-12 * (1 + expected),
                "prism distance against the oracle");

        if (0 < expected) {
          ++outside_cases;
          // Surface samples, mapped to the world with the explicit matrix
          double sampled = std::numeric_limits<double>::infinity();
          for (const auto &sp : samples)
            sampled =
                std::min(sampled, dist2_3(q, add(p.origin, mat_vec(R, sp))));
          require(d2 <= sampled + 1e-12, "prism distance below sampled");
          require(std::sqrt(d2) >= std::sqrt(sampled) - 0.3,
                  "prism distance above the sampling bound");
        } else {
          ++inside_cases;
        }
      }
    }
  require(0 < outside_cases && 0 < inside_cases,
          "prism points inside and outside");
}

// ---------------------------------------------------------------------------
// projection_coord
// ---------------------------------------------------------------------------
void test_clamped_projection_coord() {
  Rng rng(702);
  for (int iter = 0; iter < 2000; ++iter) {
    const V3 p = random_v3(rng, -3, 3);
    const V3 b = random_v3(rng, -3, 3);
    const V3 c = random_v3(rng, -3, 3);
    const V3 bc = sub(c, b);
    const double bc2 = dot3(bc, bc);
    if (bc2 < 0.1)
      continue;
    const double t = std::clamp(dot3(sub(p, b), bc) / bc2, 0., 1.);
    const V3 expected{b[0] + bc[0] * t, b[1] + bc[1] * t, b[2] + bc[2] * t};
    require(close3(from_c3(projection_coord<Coord3D, true>(to_c3(p), to_c3(b),
                                                           to_c3(c))),
                   expected, 1e-12),
            "3D clamped projection");
    const Coord2D got2 = projection_coord<Coord2D, true>(
        construct_coord_2D(p[0], p[1]), construct_coord_2D(b[0], b[1]),
        construct_coord_2D(c[0], c[1]));
    const double bc2_2 = bc[0] * bc[0] + bc[1] * bc[1];
    if (bc2_2 < 0.1)
      continue;
    const double t2 = std::clamp(
        ((p[0] - b[0]) * bc[0] + (p[1] - b[1]) * bc[1]) / bc2_2, 0., 1.);
    require(std::fabs(got2.x_ - (b[0] + bc[0] * t2)) < 1e-12 &&
                std::fabs(got2.y_ - (b[1] + bc[1] * t2)) < 1e-12,
            "2D clamped projection");
  }
  // Zero-length segment: B
  const Coord3D q = construct_coord_3D(1, 2, 3);
  require(projection_coord<Coord3D, true>(Coord3D(), q, q) == q &&
              projection_coord<Coord3D, false>(Coord3D(), q, q) == q,
          "projection onto a point");
}

void test_projection_coord() {
  Rng rng(701);
  for (int iter = 0; iter < 500; ++iter) {
    const V3 a = random_v3(rng, -3, 3);
    const V3 ab = random_v3(rng, -3, 3);
    const double ab2 = dot3(ab, ab);
    if (ab2 < 0.1)
      continue;
    // Closest point to the origin of the line through a along ab
    const V3 p = from_c3(projection_coord<Coord3D, false>(
        Coord3D(), to_c3(a), to_c3(a) + to_c3(ab)));
    const V3 expected = sub(a, scale(ab, dot3(a, ab) / ab2));
    require(close3(p, expected, 1e-12), "3D closest edge point");
    require(std::fabs(dot3(p, ab)) < 1e-12 * (1 + std::sqrt(dot3(a, a) * ab2)),
            "3D closest edge point is orthogonal to the edge");
    const V3 along = sub(a, p);
    const V3 cr{along[1] * ab[2] - along[2] * ab[1],
                along[2] * ab[0] - along[0] * ab[2],
                along[0] * ab[1] - along[1] * ab[0]};
    require(dot3(cr, cr) < 1e-20 * (1 + dot3(a, a) * ab2),
            "3D closest edge point is on the line");

    const Coord2D a2 = construct_coord_2D(a[0], a[1]);
    const Coord2D p2 = projection_coord<Coord2D, false>(
        Coord2D(), a2, a2 + construct_coord_2D(ab[0], ab[1]));
    const double ab2_2 = ab[0] * ab[0] + ab[1] * ab[1];
    if (ab2_2 < 0.1)
      continue;
    require(std::fabs(p2.x_ * ab[0] + p2.y_ * ab[1]) <
                1e-12 * (1 + std::sqrt(ab2_2)),
            "2D closest edge point is orthogonal to the edge");
    require(std::fabs((a[0] - p2.x_) * ab[1] - (a[1] - p2.y_) * ab[0]) <
                1e-12 * (1 + std::sqrt(ab2_2)),
            "2D closest edge point is on the line");
  }
}

// ---------------------------------------------------------------------------
// Shared primitives of the 2D and 3D shapes
// ---------------------------------------------------------------------------
void test_inverse_rotate_displacement_2d() {
  Rng rng(801);
  for (int iter = 0; iter < 500; ++iter) {
    const Coord2D angular = create_angular_offset(uniform(rng, -400, 400));
    const Coord2D d =
        construct_coord_2D(uniform(rng, -5, 5), uniform(rng, -5, 5));
    const Coord2D back =
        inverse_rotate_displacement(rotate_displacement(d, angular), angular);
    require(std::fabs(back.x_ - d.x_) < 1e-13 &&
                std::fabs(back.y_ - d.y_) < 1e-13,
            "2D inverse rotation round trip");
  }
  // Rotating (1, 0) places it at -90 degrees (clockwise); the inverse undoes it
  const Coord2D quarter = create_angular_offset(90);
  const Coord2D placed = rotate_displacement(construct_coord_2D(1, 0), quarter);
  require(placed.x_ == 0 && placed.y_ == -1, "clockwise quarter turn");
  require(inverse_rotate_displacement(placed, quarter) ==
              construct_coord_2D(1, 0),
          "inverse quarter turn");
}

// The support point lies on the boundary and maximizes the projection over
// boundary samples
void test_ellipsoid_support() {
  Rng rng(802);
  for (int iter = 0; iter < 300; ++iter) {
    const V3 axes = random_v3(rng, 0.1, 3);
    const Coord3D axes2 = construct_coord_3D(
        axes[0] * axes[0], axes[1] * axes[1], axes[2] * axes[2]);
    const V3 u = random_v3(rng, -1, 1);
    const Coord3D s = ellipsoid_support(axes2, to_c3(u));
    const double level = s.x_ * s.x_ / axes2.x_ + s.y_ * s.y_ / axes2.y_ +
                         s.z_ * s.z_ / axes2.z_;
    require(std::fabs(level - 1) < 1e-12, "ellipsoid support on the boundary");
    const double reach = dot3(from_c3(s), u);
    for (int k = 0; k < 200; ++k) {
      const double th = uniform(rng, 0, std::numbers::pi);
      const double ph = uniform(rng, 0, 2 * std::numbers::pi);
      const V3 e{axes[0] * std::sin(th) * std::cos(ph),
                 axes[1] * std::sin(th) * std::sin(ph), axes[2] * std::cos(th)};
      require(dot3(e, u) <= reach + 1e-12 * (1 + std::fabs(reach)),
              "ellipsoid support is the farthest point");
    }
    // 2D: the ellipse
    const Coord2D axes2_2 = construct_coord_2D(axes2.x_, axes2.y_);
    const Coord2D s2 =
        ellipsoid_support(axes2_2, construct_coord_2D(u[0], u[1]));
    require(std::fabs(s2.x_ * s2.x_ / axes2_2.x_ + s2.y_ * s2.y_ / axes2_2.y_ -
                      1) < 1e-12,
            "ellipse support on the boundary");
  }
  require(ellipsoid_support(construct_coord_3D(1, 2, 3), Coord3D()).is_null(),
          "zero direction gives the center");
}

void test_cells() {
  Rng rng(803);
  for (int iter = 0; iter < 2000; ++iter) {
    const std::array<Coord3D, 3> e3{to_c3(random_v3(rng, -2, 2)),
                                    to_c3(random_v3(rng, -2, 2)),
                                    to_c3(random_v3(rng, -2, 2))};
    const space_t det3 = cell_determinant<Coord3D>(CellEdges<Coord3D>(e3));
    require(det3 == triple_product(e3[0], e3[1], e3[2]), "3D cell determinant");
    if (std::fabs(det3) < 0.1)
      continue;
    const Coord3D p = to_c3(random_v3(rng, -3, 3));
    require(displacement_in_cell<false>(p, CellEdges<Coord3D>(e3), det3) ==
                    algebraic_projection_comparison<false>(p, e3[0], e3[1],
                                                           e3[2], det3) &&
                displacement_in_cell<true>(p, CellEdges<Coord3D>(e3), det3) ==
                    algebraic_projection_comparison<true>(p, e3[0], e3[1],
                                                          e3[2], det3),
            "3D cell forwards to the comparison");

    const std::array<Coord2D, 2> e2{construct_coord_2D(e3[0].x_, e3[0].y_),
                                    construct_coord_2D(e3[1].x_, e3[1].y_)};
    const space_t det2 = cell_determinant<Coord2D>(CellEdges<Coord2D>(e2));
    require(det2 == cross_2D(e2[0], e2[1]), "2D cell determinant");
    if (std::fabs(det2) < 0.1)
      continue;
    // Independent oracle: coefficients by the explicit inverse
    const Coord2D p2 = construct_coord_2D(p.x_, p.y_);
    const double s = (p2.x_ * e2[1].y_ - p2.y_ * e2[1].x_) / det2;
    const double t = (e2[0].x_ * p2.y_ - e2[0].y_ * p2.x_) / det2;
    const double margin = 1e-9;
    const bool in_par =
        -margin < s && s < 1 + margin && -margin < t && t < 1 + margin;
    const bool out_par =
        s < -margin || 1 + margin < s || t < -margin || 1 + margin < t;
    const bool got_par =
        displacement_in_cell<false>(p2, CellEdges<Coord2D>(e2), det2);
    require(!(in_par && !out_par) || got_par, "2D parallelogram cell inside");
    require(!out_par || !got_par, "2D parallelogram cell outside");
    const bool got_tri =
        displacement_in_cell<true>(p2, CellEdges<Coord2D>(e2), det2);
    if (s < -margin || t < -margin || 1 + margin < s + t)
      require(!got_tri, "2D triangle cell outside");
    else if (margin < s && margin < t && s + t < 1 - margin)
      require(got_tri, "2D triangle cell inside");
  }
}

void test_triangle_sample_weights() {
  Rng rng(804);
  std::uniform_real_distribution<double> dist(0., 1.);
  std::array<double, 3> mean{};
  constexpr int n = 200000;
  for (int iter = 0; iter < n; ++iter) {
    const auto w = triangle_sample_weights(dist(rng), dist(rng));
    require(0 <= w[0] && 0 <= w[1] && 0 <= w[2] &&
                std::fabs(w[0] + w[1] + w[2] - 1) < 1e-15,
            "triangle weights are barycentric");
    for (int i = 0; i < 3; ++i)
      mean[static_cast<std::size_t>(i)] += w[static_cast<std::size_t>(i)] / n;
  }
  // Uniform in the triangle: each weight has mean 1/3
  for (const double m : mean)
    require(std::fabs(m - 1. / 3) < 0.005, "triangle weights are uniform");
}

} // namespace

int main() {
  test_triple_product();
  test_create_angular_offset();
  test_rotation_general();
  test_rotation_alpha_only();
  test_rotation_right_angles();
  test_projection_comparison_lattice();
  test_projection_comparison_random();
  test_circumcenters();
  test_smallest_enclosing_ball_known();
  test_smallest_enclosing_ball_random();
  test_convex_polygon_distance2();
  test_prism_distance2();
  test_projection_coord();
  test_clamped_projection_coord();
  test_inverse_rotate_displacement_2d();
  test_ellipsoid_support();
  test_cells();
  test_triangle_sample_weights();
  std::printf("coordinate3d_test passed\n");
  return 0;
}
