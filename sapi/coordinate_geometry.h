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
#include <tuple>
#include <type_traits>
#include <vector>

#include "coordinates.h"
#include "sapi_forward_declarations.h"

namespace sapi {
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

template <Coordinate CoordT, bool clamp>
space_t projection_coeff(const CoordT &vectorBA, const CoordT &vectorBC) {
  // We project BA onto BC using orthogonal vector projection.
  const space_t dot_BA_BC = vector_dot(vectorBA, vectorBC);
  const space_t n2BC = vector_norm2(vectorBC);

  [[maybe_unused]] const bool n2_BC_az = almost_zero(n2BC);
  const bool dot_BA_BC_az = almost_zero(dot_BA_BC);
  const bool dot_BA_BC_aeq_n2BC = almost_equal(dot_BA_BC, n2BC);
  assert(!n2_BC_az || (dot_BA_BC_az && n2_BC_az));

  switch ((dot_BA_BC_az << 0) + (dot_BA_BC_aeq_n2BC << 1)) {
  case 1:
    return 0;

  case 2:
    return 1;

  case 3:
    return 0;

  default:
    // Safe to divide at this point
    if constexpr (clamp)
      return std::fmin(std::fmax(dot_BA_BC / n2BC, 0.), 1.);
    else
      return dot_BA_BC / n2BC;
  }
}

template <Coordinate CoordT, bool clamp>
inline CoordT projection_vector(const CoordT &vectorBA,
                                const CoordT &vectorBC) {
  // We project BA onto BC using orthogonal vector projection.
  return vectorBC * projection_coeff<CoordT, clamp>(vectorBA, vectorBC);
}

template <Coordinate CoordT, bool clamp>
inline CoordT projection_coord(const CoordT &coordA, const CoordT &coordB,
                               const CoordT &coordC) {
  // We project A onto BC using orthogonal vector projection.
  return coordB +
         projection_vector<CoordT, clamp>(coordA - coordB, coordC - coordB);
}

inline Coord2D create_angular_offset(const space_t angle) {
  const space_t rad_angle = radians(angle);
  return construct_coord_2D(std::cos(rad_angle), std::sin(rad_angle));
}

inline Coord2D rotate_displacement(const Coord2D &displacement,
                                   const Coord2D &angular_coord) {
  // With X = x * cos @ + y * sin @
  // and  Y = -x * sin @ + y * cos @
  return construct_coord_2D((angular_coord * displacement).sum(),
                            cross_2D(angular_coord, displacement));
}

template <Coordinate CoordT, bool triangular_comparison>
bool algebraic_projection_comparison(const CoordT &vectorBA,
                                     const CoordT &vectorBC,
                                     const CoordT &vectorBD,
                                     const space_t det_BCD) {
  // Solving comparison as equation system form
  // with pre-computed determinant
  const space_t cross_ABC = vector_cross(vectorBC, vectorBA).sum();
  const space_t cross_ABD = vector_cross(vectorBA, vectorBD).sum();

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
    // Compute 2D circumscribed center via equation system
    // to find equidistant point to all vertices
    // solved using Cramer's rule
    // Compute 3D circumscribed center by projecting 3D
    // coordinates onto 2D plane, then using 2D equation
    // system, and finally projecting back to 3D space
    c_radius.origin_ = circumscribed_center(coords);
    c_radius.radius2_ = distance2(c_radius.origin_, sorted_coords[0]);
  }

  assert(!almost_zero(c_radius.radius2_));

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

Coord2D circumscribed_center(const std::vector<Coord2D> &coords);

Coord3D circumscribed_center(const std::vector<Coord3D> &coords);

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

// Point a + ab * (t / length2) of an edge, the closest one to the origin.
// Near touching it is much shorter than a, so in 2D it is taken along the edge
// normal: its direction then carries the rounding of ab only, which GJK's
// separating-plane test needs; a + ab * t would cancel it away.
template <Coordinate CoordT>
inline CoordT closest_edge_point(const CoordT &a, const CoordT &ab,
                                 const space_t t, const space_t length2) {
  if constexpr (CoordT::D == 2) {
    const CoordT normal = construct_coord_2D(-ab.y_, ab.x_);
    return normal * (vector_dot(a, normal) / vector_norm2(normal));
  } else {
    return a + ab * (t / length2);
  }
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
  return closest_edge_point(a, ab, t, length2);
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
    const CoordT point = closest_edge_point(a, ab, d1, d1 - d3);
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
    const CoordT point = closest_edge_point(a, ac, d2, d2 - d6);
    simplex.assign(a, c);
    return point;
  }

  const space_t va = d3 * d6 - d5 * d4;
  const space_t to_c = d4 - d3;
  const space_t from_c = d5 - d6;
  if (va <= 0 && 0 <= to_c && 0 <= from_c && 0 < to_c + from_c) {
    const CoordT point = closest_edge_point(b, c - b, to_c, to_c + from_c);
    simplex.assign(b, c);
    return point;
  }

  // Face region; a degenerate (flat) triangle counts as holding the origin
  const space_t area = va + vb + vc;
  return 0 < area ? a + ab * (vb / area) + ac * (vc / area) : CoordT();
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
