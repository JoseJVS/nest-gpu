/*
 *  convex_intersection.h
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

#ifndef CONVEX_INTERSECTION_H
#define CONVEX_INTERSECTION_H

#include <cassert>
#include <concepts>
#include <type_traits>

#include "coordinate_geometry.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// Support function of a convex set: maps a direction to a farthest point of
// the set along it
template <typename SupportT, typename CoordT>
concept SupportFunction =
    Coordinate<CoordT> && std::regular_invocable<const SupportT &, CoordT> &&
    std::convertible_to<std::invoke_result_t<const SupportT &, CoordT>, CoordT>;

// Support point of the Minkowski difference A - B = {a - b : a in A, b in B}:
// s_A(u) - s_B(-u). A and B intersect iff A - B holds the origin:
//   G. van den Bergen, A Fast and Robust GJK Implementation for Collision
//   Detection of Convex Objects, Journal of Graphics Tools 4(2):7-25, 1999.
//   https://doi.org/10.1080/10867651.1999.10487502
template <Coordinate CoordT, SupportFunction<CoordT> SupportA,
          SupportFunction<CoordT> SupportB>
inline CoordT difference_support(const SupportA &support_a,
                                 const SupportB &support_b,
                                 const CoordT &direction) {
  return support_a(direction) - support_b(direction.negative());
}

// GJK (Gilbert-Johnson-Keerthi) query:
//   E. G. Gilbert, D. W. Johnson, S. S. Keerthi, A fast procedure for
//   computing the distance between complex objects in three-dimensional
//   space, IEEE Journal on Robotics and Automation 4(2):193-203, 1988.
//   https://doi.org/10.1109/56.2083
// and its boolean form (van den Bergen 1999, above).
// Whether the convex sets A and B, given by their support functions, come
// within distance margin of each other: a boolean GJK query on S = A - B,
// which comes within margin of the origin iff they do. A margin is the same
// as growing A by a ball of that radius.
//
// direction points from B towards A, e.g. the difference of interior points
// of A and B: the first simplex vertex is the support point of A - B along
// -direction. margin must not be negative. The hull of the simplex lies in
// S, so its closest point v bounds the distance from above; the support point
// w along -v bounds it from below by (v . w) / |v|. The query returns false
// only on that lower bound exceeding margin, so every uncertain path reports
// an intersection: a zero direction, NaN, no progress (duplicate or
// degenerate simplex) and max_iterations without a decision.
template <Coordinate CoordT, SupportFunction<CoordT> SupportA,
          SupportFunction<CoordT> SupportB>
bool convex_sets_intersect(const SupportA &support_a, const SupportB &support_b,
                           const CoordT &direction, const space_t margin,
                           const unsigned max_iterations = GJK_MAX_ITERATIONS) {
  if (!(0 < vector_norm2(direction)))
    return true;

  assert(0 <= margin);
  const space_t margin2 = margin * margin;
  Simplex<CoordT> simplex;
  simplex.push(difference_support(support_a, support_b, direction.negative()));
  CoordT v = simplex.points_[0];
  space_t v2 = vector_norm2(v);

  for (unsigned iteration = 0; iteration < max_iterations; ++iteration) {
    if (!(margin2 < v2))
      return true;

    const CoordT w = difference_support(support_a, support_b, v.negative());
    const space_t vw = vector_dot(v, w);
    if (0 < vw && margin2 * v2 < vw * vw)
      return false;

    simplex.push(w);
    v = closest_point_on_simplex_to_origin(simplex);
    if (simplex.size_ == CoordT::D + 1)
      return true;

    const space_t next2 = vector_norm2(v);
    if (!(next2 < v2))
      return true;
    v2 = next2;
  }
  return true;
}
} // namespace sapi

#endif
