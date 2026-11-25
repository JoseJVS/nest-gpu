/*
 *  coordinate_geometry.h
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

#ifndef COORDINATE_GEOMETRY_H
#define COORDINATE_GEOMETRY_H

#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <iterator>
#include <limits>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include "coordinates.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// 3D rotation R stored by its rows (see create_rotation_3D)
typedef std::array<Coord3D, 3> Rotation3D;

// Cells: the parallelotope (parallelogram, parallelepiped) or simplex
// (triangle, tetrahedron) spanned by D edge vectors from a corner, given as a
// span over D consecutive vectors (e.g. of helper_vectors_)
template <Coordinate CoordT>
using CellEdges = std::span<const CoordT, CoordT::D>;

template <Coordinate CoordT>
inline space_t vector_dot(const CoordT &coordA, const CoordT &coordB) {
  return (coordA * coordB).sum();
}

inline Coord2D vector_cross(const Coord2D &coordA, const Coord2D &coordB) {
  return construct_coord_2D(coordA.x_ * coordB.y_, -(coordA.y_ * coordB.x_));
}

inline Coord3D vector_cross(const Coord3D &coordA, const Coord3D &coordB) {
  return construct_coord_3D(
      vector_cross(construct_coord_2D(coordA.y_, coordB.y_),
                   construct_coord_2D(coordA.z_, coordB.z_))
          .sum(),
      -vector_cross(construct_coord_2D(coordA.x_, coordB.x_),
                    construct_coord_2D(coordA.z_, coordB.z_))
           .sum(),
      vector_cross(construct_coord_2D(coordA.x_, coordB.x_),
                   construct_coord_2D(coordA.y_, coordB.y_))
          .sum());
}

// Scalar 2D cross product (z component of the 3D cross product)
inline space_t cross_2D(const Coord2D &coordA, const Coord2D &coordB) {
  return vector_cross(coordA, coordB).sum();
}

// Twice the signed area of the triangle p, q, r: positive if counterclockwise
inline space_t orientation(const Coord2D &p, const Coord2D &q,
                           const Coord2D &r) {
  return cross_2D(q - p, r - p);
}

template <Coordinate CoordT> inline space_t vector_norm(const CoordT &coord) {
  return std::sqrt(vector_norm2(coord));
}

template <Coordinate CoordT>
inline CoordT normalize_vector(const CoordT &coord) {
  return coord.is_null() ? CoordT() : coord / vector_norm(coord);
}

template <Coordinate CoordT>
inline space_t distance2(const CoordT &coordA, const CoordT &coordB) {
  return vector_norm2(coordA - coordB);
}

template <Coordinate CoordT>
inline space_t distance(const CoordT &coordA, const CoordT &coordB) {
  return std::sqrt(distance2(coordA, coordB));
}

template <Coordinate CoordT>
inline CoordT midpoint(const CoordT &coordA, const CoordT &coordB) {
  return (coordA + coordB) / 2;
}

// Closest point to A on the line BC or, when clamped, on the segment BC (B
// when A projects before B, C after C). A minus its offset orthogonal to BC:
// along the normal of BC in 2D, BC x (BA x BC) / |BC|^2 in 3D, rather than
// BA - BC (BA . BC) / |BC|^2, which cancels when A is almost on the line. Near
// touching the offset then keeps its direction (it carries the rounding of BC
// only), which GJK's separating-plane test needs. A zero BC gives B.
template <Coordinate CoordT, bool clamp>
inline CoordT projection_coord(const CoordT &coordA, const CoordT &coordB,
                               const CoordT &coordC) {
  const CoordT vectorBA = coordA - coordB;
  const CoordT vectorBC = coordC - coordB;
  if constexpr (clamp) {
    const space_t dot_BA_BC = vector_dot(vectorBA, vectorBC);
    // Early returns: the second test is only needed after the first
    if (!(0 < dot_BA_BC))
      return coordB;
    if (!(dot_BA_BC < vector_norm2(vectorBC)))
      return coordC;
  }

  if constexpr (CoordT::D == 2) {
    const CoordT normal = construct_coord_2D(-vectorBC.y_, vectorBC.x_);
    const space_t normal2 = vector_norm2(normal);
    return 0 < normal2
               ? coordA - normal * (vector_dot(vectorBA, normal) / normal2)
               : coordB;
  } else {
    const space_t length2 = vector_norm2(vectorBC);
    return 0 < length2
               ? coordA -
                     vector_cross(vectorBC, vector_cross(vectorBA, vectorBC)) /
                         length2
               : coordB;
  }
}

// (cos, sin) of an angle in degrees, exact (0 and +-1) for the multiples of
// 90 degrees
inline Coord2D create_angular_offset(const space_t angle) {
  const bool off_axis = std::fmod(angle, 90) != 0;
  // Quarter turns (0 to 3), only used for the multiples of 90 degrees
  const int quarter =
      off_axis ? 0 : (static_cast<int>(std::fmod(angle / 90, 4)) + 4) % 4;

  // Bits 0-1: quarter turn; bit 2: not a multiple of 90 degrees
  switch ((off_axis << 2) + quarter) {
  case 0:
    return construct_coord_2D(1, 0);

  case 1:
    return construct_coord_2D(0, 1);

  case 2:
    return construct_coord_2D(-1, 0);

  case 3:
    return construct_coord_2D(0, -1);

  default: {
    const space_t rad_angle = radians(angle);
    return construct_coord_2D(std::cos(rad_angle), std::sin(rad_angle));
  }
  }
}

inline Coord2D rotate_displacement(const Coord2D &displacement,
                                   const Coord2D &angular_coord) {
  // With X = x * cos @ + y * sin @
  // and  Y = -x * sin @ + y * cos @
  return construct_coord_2D((angular_coord * displacement).sum(),
                            cross_2D(angular_coord, displacement));
}

// Inverse of the 2D rotate_displacement (the rotation by the opposite angle)
inline Coord2D inverse_rotate_displacement(const Coord2D &displacement,
                                           const Coord2D &angular_coord) {
  return rotate_displacement(
      displacement, construct_coord_2D(angular_coord.x_, -angular_coord.y_));
}

// Scalar triple product a . (b x c): the signed volume of the parallelepiped
// spanned by a, b, c (the 3x3 determinant with rows a, b, c)
inline space_t triple_product(const Coord3D &a, const Coord3D &b,
                              const Coord3D &c) {
  return vector_dot(a, vector_cross(b, c));
}

// Rotation R = Rz(alpha) Ry(beta) Rx(gamma), intrinsic z-y'-x'' angles in
// integer degrees, by rows. Multiples of 90 degrees give exact entries (0 and
// +-1).
inline Rotation3D create_rotation_3D(const angle_t alpha, const angle_t beta,
                                     const angle_t gamma) {
  const Coord2D a = create_angular_offset(alpha);
  const Coord2D b = create_angular_offset(beta);
  const Coord2D g = create_angular_offset(gamma);
  // (cos, sin) are (x_, y_)
  return {construct_coord_3D(a.x_ * b.x_, a.x_ * b.y_ * g.y_ - a.y_ * g.x_,
                             a.x_ * b.y_ * g.x_ + a.y_ * g.y_),
          construct_coord_3D(a.y_ * b.x_, a.y_ * b.y_ * g.y_ + a.x_ * g.x_,
                             a.y_ * b.y_ * g.x_ - a.x_ * g.y_),
          construct_coord_3D(-b.y_, b.x_ * g.y_, b.x_ * g.x_)};
}

// R^T d, the 3D counterpart of the 2D rotate_displacement: for a rotation
// about z alone it matches the 2D result with create_angular_offset(alpha) in
// x and y (up to rounding, exactly for multiples of 90 degrees) and keeps z.
// Like the 2D tiles and grids, a body placed with it turns clockwise.
inline Coord3D rotate_displacement(const Coord3D &displacement,
                                   const Rotation3D &rotation) {
  return rotation[0] * displacement.x_ + rotation[1] * displacement.y_ +
         rotation[2] * displacement.z_;
}

// R d, the inverse of rotate_displacement
inline Coord3D inverse_rotate_displacement(const Coord3D &displacement,
                                           const Rotation3D &rotation) {
  return construct_coord_3D(vector_dot(rotation[0], displacement),
                            vector_dot(rotation[1], displacement),
                            vector_dot(rotation[2], displacement));
}

// Whether BA = s BC + t BD has 0 <= s, t <= 1 (and s + t <= 1 for a
// triangle), by Cramer's rule with the precomputed determinant
// det_BCD = BC x BD. Closed: coefficients within almost_zero / almost_equal of
// 0 or 1 snap to them, which puts points on the edges in. The 3D overload
// below takes three coefficients from scalar triple products. Reached through
// displacement_in_cell.
template <bool triangular_comparison>
bool algebraic_projection_comparison(const Coord2D &vectorBA,
                                     const Coord2D &vectorBC,
                                     const Coord2D &vectorBD,
                                     const space_t det_BCD) {
  const space_t cross_ABC = cross_2D(vectorBC, vectorBA);
  const space_t cross_ABD = cross_2D(vectorBA, vectorBD);

  const bool cross_ABC_az = almost_zero(cross_ABC);
  const bool cross_ABC_aeq_det_BCD = almost_equal(cross_ABC, det_BCD);
  assert(!(cross_ABC_az && cross_ABC_aeq_det_BCD));

  const bool cross_ABD_az = almost_zero(cross_ABD);
  const bool cross_ABD_aeq_det_BCD = almost_equal(cross_ABD, det_BCD);
  assert(!(cross_ABD_az && cross_ABD_aeq_det_BCD));

  const space_t coeffABC =
      !cross_ABC_az *
      (!cross_ABC_aeq_det_BCD * cross_ABC / det_BCD + cross_ABC_aeq_det_BCD);
  const space_t coeffABD =
      !cross_ABD_az *
      (!cross_ABD_aeq_det_BCD * cross_ABD / det_BCD + cross_ABD_aeq_det_BCD);

  switch ((cross_ABC_az << 0) + (cross_ABD_az << 1) +
          (cross_ABC_aeq_det_BCD << 2) + (cross_ABD_aeq_det_BCD << 3)) {
  case 1:
    return std::isless(0, coeffABD) && std::isless(coeffABD, 1);

  case 2:
    return std::isless(0, coeffABC) && std::isless(coeffABC, 1);

  case 3:
    return true;

  case 4:
    return !triangular_comparison && std::isless(0, coeffABD) &&
           std::isless(coeffABD, 1);

  case 8:
    return !triangular_comparison && std::isless(0, coeffABC) &&
           std::isless(coeffABC, 1);

  case 12:
    return !triangular_comparison;

  case 6:
    return true;

  case 9:
    return true;

  default:
    return std::isless(0, coeffABC) && std::isless(coeffABC, 1) &&
           std::isless(0, coeffABD) && std::isless(coeffABD, 1) &&
           leq_test(coeffABC + coeffABD, 2 - triangular_comparison);
  }
}

// Cramer coefficient numerator / det, snapped to 0 or 1 when the numerator is
// almost zero or almost equal to det (as in the 2D comparison)
inline space_t snapped_cramer_coefficient(const space_t numerator,
                                          const space_t det) {
  switch ((almost_zero(numerator) << 0) + (almost_equal(numerator, det) << 1)) {
  case 1:
    return 0;

  case 2:
    return 1;

  case 3:
    return 0;

  default:
    return numerator / det;
  }
}

// Whether BA = s BC + t BD + u BE has 0 <= s, t, u <= 1 (the parallelepiped
// spanned by BC, BD, BE from B) and, for a tetrahedron, s + t + u <= 1 (the
// tetrahedron B, C, D, E). Closed: the snapped coefficients put points on the
// faces in.
template <bool tetrahedral_comparison>
bool algebraic_projection_comparison(const Coord3D &vectorBA,
                                     const Coord3D &vectorBC,
                                     const Coord3D &vectorBD,
                                     const Coord3D &vectorBE,
                                     const space_t det_BCDE) {
  // Coefficients (s, t, u) by Cramer's rule with scalar triple products
  const Coord3D coeffs = construct_coord_3D(
      snapped_cramer_coefficient(triple_product(vectorBA, vectorBD, vectorBE),
                                 det_BCDE),
      snapped_cramer_coefficient(triple_product(vectorBC, vectorBA, vectorBE),
                                 det_BCDE),
      snapped_cramer_coefficient(triple_product(vectorBC, vectorBD, vectorBA),
                                 det_BCDE));
  const bool in_unit_cube =
      std::islessequal(0, coeffs.x_) && std::islessequal(coeffs.x_, 1) &&
      std::islessequal(0, coeffs.y_) && std::islessequal(coeffs.y_, 1) &&
      std::islessequal(0, coeffs.z_) && std::islessequal(coeffs.z_, 1);
  return in_unit_cube && (!tetrahedral_comparison || leq_test(coeffs.sum(), 1));
}

// Signed volume of the parallelotope spanned by the edges: cross_2D in 2D,
// triple_product in 3D
template <Coordinate CoordT>
inline space_t cell_determinant(const CellEdges<CoordT> edges) {
  if constexpr (CoordT::D == 2)
    return cross_2D(edges[0], edges[1]);
  else
    return triple_product(edges[0], edges[1], edges[2]);
}

// Whether the displacement from the corner lies in the closed parallelotope
// (simplex: in the simplex) spanned by the edges; det is their
// cell_determinant
template <bool simplex, Coordinate CoordT>
inline bool displacement_in_cell(const CoordT &displacement,
                                 const CellEdges<CoordT> edges,
                                 const space_t det) {
  if constexpr (CoordT::D == 2)
    return algebraic_projection_comparison<simplex>(displacement, edges[0],
                                                    edges[1], det);
  else
    return algebraic_projection_comparison<simplex>(displacement, edges[0],
                                                    edges[1], edges[2], det);
}

// Barycentric weights (w0, w1, w2) of a uniform point in a triangle from two
// uniform draws m0, m1 in [0, 1]: draws with m0 + m1 > 1 fold back into the
// triangle
inline std::array<space_t, 3> triangle_sample_weights(space_t m0, space_t m1) {
  const bool adjust = std::isless(1., m0 + m1);
  m0 = adjust * std::fmax(std::fmin(1., 1. - m0), 0.) + !adjust * m0;
  m1 = adjust * std::fmax(std::fmin(1., 1. - m1), 0.) + !adjust * m1;
  return {m0, m1, std::fmax(std::fmin(1., 1. - m0 - m1), 0.)};
}

// Center of the circle through a, b, c in their plane (any dimension), from
// the Gram system of the edges u = b - a and v = c - a (relative to a, so it
// does not lose precision far from the origin). Not finite for collinear
// points.
template <Coordinate CoordT>
inline CoordT circumscribed_center(const CoordT &a, const CoordT &b,
                                   const CoordT &c) {
  const CoordT u = b - a;
  const CoordT v = c - a;
  const space_t uu = vector_norm2(u);
  const space_t vv = vector_norm2(v);
  const space_t uv = vector_dot(u, v);
  const space_t det2 = 2 * (uu * vv - uv * uv);
  return a + u * (vv * (uu - uv) / det2) + v * (uu * (vv - uv) / det2);
}

// Center of the sphere through a, b, c, d: with u, v, w the edges from a,
//   a + (|u|^2 v x w + |v|^2 w x u + |w|^2 u x v) / (2 u . (v x w)).
// Not finite for coplanar points.
inline Coord3D circumscribed_center(const Coord3D &a, const Coord3D &b,
                                    const Coord3D &c, const Coord3D &d) {
  const Coord3D u = b - a;
  const Coord3D v = c - a;
  const Coord3D w = d - a;
  return a + (vector_cross(v, w) * vector_norm2(u) +
              vector_cross(w, u) * vector_norm2(v) +
              vector_cross(u, v) * vector_norm2(w)) /
                 (2 * triple_product(u, v, w));
}

template <Coordinate CoordT>
std::tuple<vertidx_t, std::array<CoordT, 3>, std::array<CoordT, 3>,
           std::array<space_t, 3>>
sort_triangular_vertices(const std::vector<CoordT> &coord_in) {
  assert(coord_in.size() == 3);

  vertidx_t A_vix = 0, B_vix = 0, C_vix = 0, vix_next = 0, equal_lengths = 0;
  space_t largest_length = 0, prev_length = 0;

  CoordT indexed_vectors[3];
  space_t indexed_lengths[3];
  bool almost_equal_lengths[3] = {false};
  for (vertidx_t vix = 0; vix < 3; ++vix) {
    vix_next = static_cast<vertidx_t>((vix + 1) % 3);
    indexed_vectors[vix] = coord_in[vix_next] - coord_in[vix];
    indexed_lengths[vix] = vector_norm2(indexed_vectors[vix]);
    assert(!almost_zero(indexed_lengths[vix]));

    if (0 < vix && almost_equal(prev_length, indexed_lengths[vix])) {
      almost_equal_lengths[vix - 1] = true;
      ++equal_lengths;
    } else if (0 == vix || std::isless(largest_length, indexed_lengths[vix])) {
      A_vix = vix;
      B_vix = vix_next;
      largest_length = indexed_lengths[vix];
    }

    prev_length = indexed_lengths[vix];
  }
  almost_equal_lengths[2] = almost_equal(prev_length, indexed_lengths[0]);
  equal_lengths += almost_equal_lengths[2];
  C_vix = static_cast<vertidx_t>((B_vix + 1) % 3);

  // Due to commutative comparison equal lengths == 2 implies equal lengths == 3
  const bool all_lengths_equal = equal_lengths == 2;
  equal_lengths += all_lengths_equal;

  // A triangle either starts as a right triangle from a square
  // i.e. isosceles right triangle or an equilateral triangle.
  // When spliting an isosceles right triangle, symmetry is preserved
  // and resulting triangles are isosceles right triangles too.
  // When splitting an equilateral triangle resulting triangles
  // are two right triangles with no equal lengths.
  // For right triangles with no equal lengths we return
  // the hypotenuse vertices and apex vertex in that order.
  // For an equilateral triangle any combination is fine.
  // For an isosceles triangle (right or not) we return the base vertices
  // and apex vertex in that order.
  if (equal_lengths == 1) {
    // Get end vertex of base edge of isosceles triangle
    for (B_vix = 0; B_vix < 3; ++B_vix)
      if (almost_equal_lengths[B_vix])
        break;

    // AB edge is base of isosceles
    A_vix = static_cast<vertidx_t>((B_vix + 2) % 3);
    // CB edge is one of the two equal sides
    C_vix = static_cast<vertidx_t>((B_vix + 1) % 3);
  }

  return std::make_tuple(
      equal_lengths,
      std::array<CoordT, 3>{coord_in[A_vix], coord_in[B_vix], coord_in[C_vix]},
      std::array<CoordT, 3>{indexed_vectors[A_vix], indexed_vectors[B_vix],
                            indexed_vectors[C_vix]},
      std::array<space_t, 3>{indexed_lengths[A_vix], indexed_lengths[B_vix],
                             indexed_lengths[C_vix]});
}

template <Coordinate CoordT>
CircumscribedRadius<CoordT>
compute_c_radius(const std::vector<CoordT> &coords) {
  assert(coords.size() == 3);

  CircumscribedRadius<CoordT> c_radius;

  const auto [equal_lengths, sorted_coords, sorted_vectors, sorted_lengths] =
      sort_triangular_vertices(coords);

  const CoordT base_vertices_add(sorted_coords[0] + sorted_coords[1]);

  if (equal_lengths == 3) {
    // Implies equilateral triangle, circumscribed center is
    // a scaled orthogonal projection of apex vertex onto base edge
    // here apex vertex and base edge are an arbitrary choice

    // With side length a, radius r is equal to a / sqrt( 3 )
    c_radius.radius2_ = sorted_lengths[0] / 3;

    // With side length a, height h is equal to sqrt( 3 ) * a / 2
    // projection coefficient of apex vertex onto base edge is r / h == 2 / 3
    // Using vector combining formula of apex + ( midlepoint of base edge - apex
    // ) * coefficient we reduce to apex * ( 1 - coefficient ) + ( base edge *
    // coefficient / 2 ) with coefficient == 2 / 3 we have apex / 3 + base edge
    // / 3
    c_radius.origin_ = (sorted_coords[2] / 3) + (base_vertices_add / 3);
  } else if (almost_zero(vector_dot(sorted_vectors[1], sorted_vectors[2]))) {
    // Right triangle case, circumscribed center is the midpoint of the
    // hypotenuse hypotenuse == base and radius == | base | / 2
    c_radius.origin_ = base_vertices_add / 2;
    c_radius.radius2_ = sorted_lengths[0] / 4;
  } else if (equal_lengths == 1) {
    // Plain isosceles case, circumscribed center is
    // a scaled orthogonal projection of apex vertex onto base edge

    // With base length b and side length a
    // height h is equal to sqrt( a2 - ( b2 / 4 ) )
    // => 4 * h2 = 4 * a2 - b2
    const space_t height2_4 =
        std::fma(4., sorted_lengths[2], -sorted_lengths[0]);

    // With base height h and side length a
    // radius r is equal to a2 / ( 2 * h )
    //  => radius2 = a4 / ( 4 * h2 )
    c_radius.radius2_ = (sorted_lengths[2] * sorted_lengths[2]) / height2_4;

    // Projection coefficient of apex vertex onto base edge is r / h
    const space_t projection_coefficient = sorted_lengths[2] / (height2_4 / 2);

    c_radius.origin_ = sorted_coords[2] * (1. - projection_coefficient) +
                       base_vertices_add * (projection_coefficient / 2);
  } else {
    // General triangle: the circumcenter from the Gram system of the edges
    c_radius.origin_ = circumscribed_center(coords[0], coords[1], coords[2]);
    c_radius.radius2_ = distance2(c_radius.origin_, sorted_coords[0]);
  }

  assert(std::isfinite(c_radius.radius2_) && !almost_zero(c_radius.radius2_));

  return c_radius;
}

template <Coordinate CoordT>
inline void update_min(CoordT &min, const CoordT &value) {
  min.x_ = std::isless(value.x_, min.x_) ? value.x_ : min.x_;
  min.y_ = std::isless(value.y_, min.y_) ? value.y_ : min.y_;

  if constexpr (std::is_same_v<std::remove_cvref_t<CoordT>, Coord3D>)
    min.z_ = std::isless(value.z_, min.z_) ? value.z_ : min.z_;
}

template <Coordinate CoordT>
inline void update_max(CoordT &max, const CoordT &value) {
  max.x_ = std::isless(max.x_, value.x_) ? value.x_ : max.x_;
  max.y_ = std::isless(max.y_, value.y_) ? value.y_ : max.y_;

  if constexpr (std::is_same_v<std::remove_cvref_t<CoordT>, Coord3D>)
    max.z_ = std::isless(max.z_, value.z_) ? value.z_ : max.z_;
}

template <Coordinate CoordT> inline std::pair<CoordT, CoordT> minmax_coords() {

  CoordT min, max;
  min.x_ = std::numeric_limits<space_t>::max();
  min.y_ = std::numeric_limits<space_t>::max();
  max.x_ = std::numeric_limits<space_t>::lowest();
  max.y_ = std::numeric_limits<space_t>::lowest();

  if constexpr (std::is_same_v<std::remove_cvref_t<CoordT>, Coord3D>) {
    min.z_ = std::numeric_limits<space_t>::max();
    max.z_ = std::numeric_limits<space_t>::lowest();
  }

  return std::make_pair(min, max);
}

template <Coordinate CoordT>
std::pair<CoordT, CoordT>
minmax_vertices(const std::vector<CoordT> &coord_vec) {
  assert(1 < coord_vec.size());

  auto min_max = minmax_coords<CoordT>();

  for (const auto &coord : coord_vec) {
    update_min(min_max.first, coord);
    update_max(min_max.second, coord);
  }

  return min_max;
}

// Largest squared distance from the point to the points. The point of a convex
// polytope farthest from any point is a vertex, so over the vertices it is the
// polytope's.
template <Coordinate CoordT, typename PointRange>
inline space_t max_distance2(const PointRange &points, const CoordT &point) {
  space_t max_d2 = 0.;
  for (const auto &p : points)
    max_d2 = std::fmax(max_d2, distance2(p, point));
  return max_d2;
}

// Largest squared norm of the differences b - a - offset: over the vertices of
// two convex polytopes A and B, that of the convex set B - A - offset
template <Coordinate CoordT, typename PointRange>
inline space_t max_difference_norm2(const PointRange &a, const PointRange &b,
                                    const CoordT &offset) {
  space_t max_d2 = 0.;
  for (const auto &p : a)
    max_d2 = std::fmax(max_d2, max_distance2(b, p + offset));
  return max_d2;
}

// Smallest ball (disk in 2D) holding at most D + 1 points: the smallest of the
// balls through 1 to D + 1 of them (each with those points on its boundary)
// that holds the others within leq_test. Degenerate subsets (collinear,
// coplanar) give no finite center and are skipped; a smaller subset covers
// them. The radius is the largest distance to the chosen center, so every
// point lies in the ball as computed.
template <Coordinate CoordT, typename PointRange>
CircumscribedRadius<CoordT> smallest_enclosing_ball(const PointRange &points) {
  std::array<CoordT, CoordT::D + 1> p{};
  std::size_t n = 0;
  for (const auto &point : points) {
    assert(n < p.size());
    p[n++] = point;
  }
  assert(0 < n);

  CircumscribedRadius<CoordT> best{
      .radius2_ = std::numeric_limits<space_t>::infinity(), .origin_ = p[0]};
  for (unsigned subset = 1; subset < (1U << n); ++subset) {
    std::array<CoordT, CoordT::D + 1> s{};
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i)
      if (subset & (1U << i))
        s[k++] = p[i];

    CoordT center;
    switch (k) {
    case 1:
      center = s[0];
      break;

    case 2:
      center = midpoint(s[0], s[1]);
      break;

    case 3:
      center = circumscribed_center(s[0], s[1], s[2]);
      break;

    default:
      if constexpr (CoordT::D == 3)
        center = circumscribed_center(s[0], s[1], s[2], s[3]);
      break;
    }

    const space_t radius2 = distance2(center, s[0]);
    if (!std::isfinite(radius2) || !std::isless(radius2, best.radius2_))
      continue;

    bool holds_all = true;
    for (std::size_t i = 0; i < n && holds_all; ++i)
      holds_all = leq_test(distance2(center, p[i]), radius2);
    if (holds_all)
      best = {.radius2_ = radius2, .origin_ = center};
  }

  best.radius2_ =
      max_distance2(std::span<const CoordT>(p.data(), n), best.origin_);
  return best;
}

// Closest point to coord on the perimeter of a convex polygon, the vertices in
// perimeter order: every edge is projected and the closest result is kept.
// Each edge is projected from its nearer endpoint (the lower index on ties); a
// vertex within almost_zero is returned as is. Out of line: it is too large to
// inline into the translation units of the per-pair loops.
Coord2D
project_point_to_polygon_perimeter(const Coord2D &coord,
                                   const std::span<const Coord2D> &vertices);

// Squared distance from a point to a convex polygon (0 inside or on it), the
// vertices in counterclockwise order
template <typename PointRange>
space_t convex_polygon_distance2(const Coord2D &point,
                                 const PointRange &vertices) {
  auto prev = std::prev(std::end(vertices));
  for (auto it = std::begin(vertices); it != std::end(vertices); prev = it++)
    if (std::isless(orientation(*prev, *it, point), 0))
      return distance2(point,
                       project_point_to_polygon_perimeter(point, vertices));
  return 0;
}

// Squared distance from a point to a right prism, the convex polygon of the
// face (vertices in counterclockwise order) times the interval [bottom, top]
// along the axis, from the point in the prism frame (x, y in the face plane, z
// along the axis): the squared distance to the face polygon plus the squared
// distance of the axial coordinate to the interval
template <typename PointRange>
inline space_t prism_distance2(const Coord3D &point,
                               const PointRange &face_vertices,
                               const space_t bottom, const space_t top) {
  const space_t axial_gap =
      std::fmax(std::fmax(bottom - point.z_, point.z_ - top), 0);
  return convex_polygon_distance2(construct_coord_2D(point.x_, point.y_),
                                  face_vertices) +
         axial_gap * axial_gap;
}

// Point of points farthest along direction (the first one on ties): the support
// point of their convex hull
template <Coordinate CoordT, typename PointRange>
inline CoordT farthest_point(const PointRange &points,
                             const CoordT &direction) {
  assert(std::begin(points) != std::end(points));
  auto best = std::begin(points);
  space_t best_dot = vector_dot(*best, direction);
  for (auto it = std::next(best); it != std::end(points); ++it) {
    const space_t dot = vector_dot(*it, direction);
    if (best_dot < dot) {
      best_dot = dot;
      best = it;
    }
  }
  return *best;
}

// Closest points to the origin of the simplex hull. Each reduces the simplex
// to the vertices of the feature holding that point. Degenerate features
// (duplicate or collinear vertices) never divide by zero; they may keep a
// larger feature than needed.
template <Coordinate CoordT>
inline CoordT closest_point_on_segment_to_origin(Simplex<CoordT> &simplex) {
  // References into the simplex: every result is taken before assign()
  // overwrites the vertices
  const CoordT &a = simplex.points_[0];
  const CoordT &b = simplex.points_[1];
  const CoordT ab = b - a;
  // Projection of the origin on the segment line, scaled by |ab|^2
  const space_t t = -vector_dot(a, ab);
  if (!(0 < t)) {
    simplex.assign(a);
    return simplex.points_[0];
  }
  const space_t length2 = vector_norm2(ab);
  if (!(t < length2)) {
    simplex.assign(b);
    return simplex.points_[0];
  }
  // Interior: the callers classify the endpoint regions, since GJK reduces
  // the simplex to the feature holding the point
  return projection_coord<CoordT, false>(CoordT(), a, b);
}

// Voronoi (feature) regions of the triangle with only dot products, so it
// holds in any dimension. In 2D the face region holds the origin; in 3D the
// result is its projection on the plane:
//   C. Ericson, Real-Time Collision Detection, Morgan Kaufmann, 2005,
//   section 5.1.5 "Closest Point on Triangle to Point".
//   https://doi.org/10.1201/b14581
template <Coordinate CoordT>
CoordT closest_point_on_triangle_to_origin(Simplex<CoordT> &simplex) {
  // References into the simplex: every result is taken before assign()
  // overwrites the vertices
  const CoordT &a = simplex.points_[0];
  const CoordT &b = simplex.points_[1];
  const CoordT &c = simplex.points_[2];
  const CoordT ab = b - a;
  const CoordT ac = c - a;

  const space_t d1 = -vector_dot(ab, a);
  const space_t d2 = -vector_dot(ac, a);
  if (d1 <= 0 && d2 <= 0) {
    simplex.assign(a);
    return simplex.points_[0];
  }

  const space_t d3 = -vector_dot(ab, b);
  const space_t d4 = -vector_dot(ac, b);
  if (0 <= d3 && d4 <= d3) {
    simplex.assign(b);
    return simplex.points_[0];
  }

  const space_t vc = d1 * d4 - d3 * d2;
  // Edge regions; a degenerate edge (zero length) claims none
  if (vc <= 0 && 0 <= d1 && d3 <= 0 && 0 < d1 - d3) {
    const CoordT point = projection_coord<CoordT, false>(CoordT(), a, b);
    simplex.assign(a, b);
    return point;
  }

  const space_t d5 = -vector_dot(ab, c);
  const space_t d6 = -vector_dot(ac, c);
  if (0 <= d6 && d5 <= d6) {
    simplex.assign(c);
    return simplex.points_[0];
  }

  const space_t vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && 0 <= d2 && d6 <= 0 && 0 < d2 - d6) {
    const CoordT point = projection_coord<CoordT, false>(CoordT(), a, c);
    simplex.assign(a, c);
    return point;
  }

  const space_t va = d3 * d6 - d5 * d4;
  const space_t to_c = d4 - d3;
  const space_t from_c = d5 - d6;
  if (va <= 0 && 0 <= to_c && 0 <= from_c && 0 < to_c + from_c) {
    const CoordT point = projection_coord<CoordT, false>(CoordT(), b, c);
    simplex.assign(b, c);
    return point;
  }

  // Face region; a degenerate (flat) triangle counts as holding the origin.
  // In 3D the point is taken along the plane normal, which keeps its
  // direction near touching (as projection_coord does for edges).
  const space_t area = va + vb + vc;
  if (!(0 < area))
    return CoordT();
  if constexpr (CoordT::D == 3) {
    const CoordT normal = vector_cross(ab, ac);
    const space_t normal2 = vector_norm2(normal);
    return 0 < normal2 ? normal * (vector_dot(a, normal) / normal2) : CoordT();
  } else {
    return a + ab * (vb / area) + ac * (vc / area);
  }
}

// Keeps the full tetrahedron when it holds the origin (also when it is flat),
// otherwise the face feature closest to the origin (Ericson 2005, above,
// section 5.1.6 "Closest Point on Tetrahedron to Point")
Coord3D closest_point_on_tetrahedron_to_origin(Simplex<Coord3D> &simplex);

// Closest point to the origin of the simplex hull, reducing the simplex to
// the vertices of the feature holding it. The simplex keeps D + 1 vertices
// only when its hull holds the origin (or it is degenerate and may hold it).
template <Coordinate CoordT>
inline CoordT closest_point_on_simplex_to_origin(Simplex<CoordT> &simplex) {
  switch (simplex.size_) {
  case 1:
    return simplex.points_[0];
  case 2:
    return closest_point_on_segment_to_origin(simplex);
  case 3:
    return closest_point_on_triangle_to_origin(simplex);
  default:
    if constexpr (CoordT::D == 3) {
      return closest_point_on_tetrahedron_to_origin(simplex);
    } else {
      assert(false && "A 2D simplex has at most three vertices");
      return CoordT();
    }
  }
}

// Support point of the axis-aligned ellipse (ellipsoid in 3D) centered at
// the origin with squared semi-axes semi_axes2, along direction:
// semi_axes2 q / sqrt(sum(semi_axes2 q^2)) (the unit-sphere support under the
// affine map); the center for a zero direction
template <Coordinate CoordT>
inline CoordT ellipsoid_support(const CoordT &semi_axes2,
                                const CoordT &direction) {
  const space_t norm = std::sqrt((semi_axes2 * direction * direction).sum());
  if (!(0 < norm))
    return CoordT();
  return semi_axes2 * direction / norm;
}

// Given ellipse with h, k center, a semi major, b semi minor, @ rotation
// If x, y point is in ellipse then
// X^2 / a^2 + Y^2 / b^2 <= 1
// With X = ( x - h ) * cos @ + ( y - k ) * sin @
// and  Y = - ( x - h ) * sin @ + ( y - k ) * cos @
inline bool displacement_in_ellipse(const Coord2D &displacement,
                                    const Coord2D &semi_axes) {
  switch ((almost_zero(displacement.x_) << 0) +
          (almost_zero(displacement.y_) << 1)) {
  case 3:
    return true;

  case 1:
    return leq_test(displacement.y_ * displacement.y_, semi_axes.y_);

  case 2:
    return leq_test(displacement.x_ * displacement.x_, semi_axes.x_);

  default:
    return leq_test(semi_axes.y_ * displacement.x_ * displacement.x_ +
                        semi_axes.x_ * displacement.y_ * displacement.y_,
                    semi_axes.product());
  }
}

inline bool displacement_in_ellipsoid(const Coord3D &displacement,
                                      const Coord3D &semi_axes) {
  switch ((almost_zero(displacement.x_) << 0) +
          (almost_zero(displacement.y_) << 1) +
          (almost_zero(displacement.z_) << 2)) {
  case 7:
    return true;

  case 1:
    return leq_test(semi_axes.z_ * displacement.y_ * displacement.y_ +
                        semi_axes.y_ * displacement.z_ * displacement.z_,
                    semi_axes.y_ * semi_axes.z_);

  case 2:
    return leq_test(semi_axes.z_ * displacement.x_ * displacement.x_ +
                        semi_axes.x_ * displacement.z_ * displacement.z_,
                    semi_axes.x_ * semi_axes.z_);

  case 4:
    return leq_test(semi_axes.y_ * displacement.x_ * displacement.x_ +
                        semi_axes.x_ * displacement.y_ * displacement.y_,
                    semi_axes.x_ * semi_axes.y_);

  case 3:
    return leq_test(displacement.z_ * displacement.z_, semi_axes.z_);

  case 5:
    return leq_test(displacement.y_ * displacement.y_, semi_axes.y_);

  case 6:
    return leq_test(displacement.x_ * displacement.x_, semi_axes.x_);

  default:
    return leq_test(
        semi_axes.y_ * semi_axes.z_ * displacement.x_ * displacement.x_ +
            semi_axes.x_ * semi_axes.z_ * displacement.y_ * displacement.y_ +
            semi_axes.x_ * semi_axes.y_ * displacement.z_ * displacement.z_,
        semi_axes.product());
  }
}
} // namespace sapi

#endif
