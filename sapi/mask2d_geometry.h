/*
 *  mask2d_geometry.h
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

#ifndef MASK2D_GEOMETRY_H
#define MASK2D_GEOMETRY_H

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include "convex_intersection.h"
#include "coordinate_geometry.h"
#include "mask.h"
#include "sapi_forward_declarations.h"
#include "tile.h"

namespace sapi {
// Shape helpers of 2D masks, defined in mask2d_geometry.cpp; each checks its
// own parameters
void initialize_circular_mask_helpers(Mask<Coord2D> &mask,
                                      const std::vector<space_t> &mask_params);

void initialize_elliptical_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

void initialize_parallelogram_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

void initialize_triangular_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params);

void initialize_annular_mask_helpers(Mask<Coord2D> &mask,
                                     const std::vector<space_t> &mask_params);

// Centered shapes (circle, ellipse, parallelogram): the bounding disk is
// centered on the anchor, so the disk displacement is the anchor displacement
inline OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &coord,
                                               const Mask<Coord2D> &mask) {
  return mask.bounding_disk_.coord_in_radius(coord);
}

inline OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &a,
                                               const Coord2D &b,
                                               const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  od.first = leq_test(od.second.distance2_, mask.bounding_disk_.radius2_);
  return od;
}

inline OptDisp<Coord2D> coord_in_elliptical_mask(const Coord2D &coord,
                                                 const Mask<Coord2D> &mask) {
  auto od = mask.bounding_disk_.coord_in_radius(coord);
  if (od.first)
    od.first = displacement_in_ellipse(
        1 < mask.helper_vectors_.size()
            ? rotate_displacement(od.second.displacement_,
                                  mask.helper_vectors_[1])
            : od.second.displacement_,
        mask.helper_vectors_[0]);
  return od;
}

inline OptDisp<Coord2D> coord_in_elliptical_mask(const Coord2D &a,
                                                 const Coord2D &b,
                                                 const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  if (leq_test(od.second.distance2_, mask.bounding_disk_.radius2_))
    od.first = displacement_in_ellipse(
        1 < mask.helper_vectors_.size()
            ? rotate_displacement(od.second.displacement_,
                                  mask.helper_vectors_[1])
            : od.second.displacement_,
        mask.helper_vectors_[0]);
  return od;
}

inline OptDisp<Coord2D> coord_in_parallelogram_mask(const Coord2D &coord,
                                                    const Mask<Coord2D> &mask) {
  auto od = mask.bounding_disk_.coord_in_radius(coord);
  if (od.first)
    od.first = algebraic_projection_comparison<false>(
        od.second.displacement_ + mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_vectors_[2],
        mask.helper_scalars_[0]);

  return od;
}

inline OptDisp<Coord2D> coord_in_parallelogram_mask(const Coord2D &a,
                                                    const Coord2D &b,
                                                    const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  if (leq_test(od.second.distance2_, mask.bounding_disk_.radius2_))
    od.first = algebraic_projection_comparison<false>(
        od.second.displacement_ + mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_vectors_[2],
        mask.helper_scalars_[0]);

  return od;
}

// The triangle is defined relative to its anchor vertex while its bounding
// disk is centered on the triangle: the prefilter uses the disk, the exact test
// and the returned displacement use the anchor
inline OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &coord,
                                                 const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(coord - mask.anchor_));
  if (mask.bounding_disk_.coord_in_radius(coord).first)
    od.first = algebraic_projection_comparison<true>(
        od.second.displacement_, mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_scalars_[0]);

  return od;
}

inline OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &a,
                                                 const Coord2D &b,
                                                 const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  if (leq_test(vector_norm2(b - (a + mask.bounding_disk_.origin_)),
               mask.bounding_disk_.radius2_))
    od.first = algebraic_projection_comparison<true>(
        od.second.displacement_, mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_scalars_[0]);

  return od;
}

// Annulus: helper_scalars_[0] holds the squared inner radius; the outer
// circle is the bounding disk, centered on the anchor. Closed ring: within the
// outer radius and not inside the inner one. The inner test is exact (no
// leq_test tolerance); the tile filters only reject with a margin
inline bool displacement_in_annulus(const space_t distance2,
                                    const Mask<Coord2D> &mask) {
  return leq_test(distance2, mask.bounding_disk_.radius2_) &&
         !(distance2 < mask.helper_scalars_[0]);
}

inline OptDisp<Coord2D> coord_in_annular_mask(const Coord2D &coord,
                                              const Mask<Coord2D> &mask) {
  auto od = mask.bounding_disk_.coord_in_radius(coord);
  od.first = od.first && !(od.second.distance2_ < mask.helper_scalars_[0]);
  return od;
}

inline OptDisp<Coord2D> coord_in_annular_mask(const Coord2D &a,
                                              const Coord2D &b,
                                              const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  od.first = displacement_in_annulus(od.second.distance2_, mask);
  return od;
}

// Support point of the ellipse centered at the origin: in the frame of its
// axes, (a^2 q_x, b^2 q_y) / sqrt(a^2 q_x^2 + b^2 q_y^2) for the direction q
// (the unit-circle support under the affine map), rotated back. The center
// for a zero direction. helper_vectors_[0] holds the squared semi-axes
// (a^2, b^2) and, when rotated, helper_vectors_[1] the (cos, sin) of the
// rotation.
inline Coord2D ellipse_support(const Mask<Coord2D> &mask,
                               const Coord2D &direction) {
  const bool rotated = 1 < mask.helper_vectors_.size();
  const Coord2D q =
      rotated ? rotate_displacement(direction, mask.helper_vectors_[1])
              : direction;
  const space_t norm = std::sqrt((mask.helper_vectors_[0] * q * q).sum());
  if (!(0 < norm))
    return Coord2D();

  const Coord2D point = mask.helper_vectors_[0] * q / norm;
  if (!rotated)
    return point;
  return rotate_displacement(point,
                             construct_coord_2D(mask.helper_vectors_[1].x_,
                                                -mask.helper_vectors_[1].y_));
}

// Support point of the mask for GJK: its farthest point along direction, in
// the frame of anchor_ (absolute for source/target masks, an offset from the
// driver node for blueprints). Circles and the annulus have exact closed-form
// tile tests and never reach GJK. Vertices relative to the anchor:
// parallelogram a0, a1, -a0, -a1 (helper_vectors_[0] and [3]); triangle 0 (the
// anchor), a0, a1 (helper_vectors_[0] and [1]).
inline Coord2D mask_support(const Mask<Coord2D> &mask,
                            const Coord2D &direction) {
  switch (mask.shape_) {
  case MASK_SHAPE::ELLIPTICAL:
    return mask.anchor_ + ellipse_support(mask, direction);
  case MASK_SHAPE::PARALLELOGRAM:
    return mask.anchor_ +
           farthest_point(
               std::array<Coord2D, 4>{mask.helper_vectors_[0],
                                      mask.helper_vectors_[3],
                                      mask.helper_vectors_[0].negative(),
                                      mask.helper_vectors_[3].negative()},
               direction);
  case MASK_SHAPE::TRIANGULAR:
    return mask.anchor_ +
           farthest_point(std::array<Coord2D, 3>{Coord2D(),
                                                 mask.helper_vectors_[0],
                                                 mask.helper_vectors_[1]},
                          direction);
  default:
    assert(false && "Invalid mask shape");
    return mask.anchor_;
  }
}

// Tile tests per shape, dispatched from mask_geometry.h after the generic fast
// tests of Mask. A circle equals its bounding disk, so the disk decides it
// exactly; the other shapes use the exact intersection (GJK) of their support
// with the tiles, within exact_filter_margin.

// Shared by the convex shapes: a mask holding every vertex of a tile holds
// the tile
inline OVERLAP_LEVEL
overlap_with_all_vertices_in_convex_mask(const Mask<Coord2D> &,
                                         const Tile<Coord2D> &) {
  return OVERLAP_LEVEL::FULL;
}

// A convex mask contains its anchor, so a tile holding the anchor meets the
// mask (the cheap accept the tests below start with). The annulus does not
// contain its anchor (the center): it reaches this accept through the circle
// tests only after its hole reject, which makes it valid there too.
inline bool anchor_in_tile(const Mask<Coord2D> &mask,
                           const Tile<Coord2D> &tile) {
  return tile.coord_in_tile(mask.anchor_);
}

// Blueprint placed at the node coord
inline bool anchor_in_tile(const Mask<Coord2D> &mask, const Coord2D &coord,
                           const Tile<Coord2D> &tile) {
  return tile.coord_in_tile(coord + mask.anchor_);
}

// Source/target masks: the tile perimeter point closest to the center lies
// within the radius
inline bool circular_mask_meets_tile(const Mask<Coord2D> &mask,
                                     const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, tile))
    return true;

  return mask.bounding_disk_
      .coord_in_radius(
          tile.project_point_to_surface(mask.bounding_disk_.origin_))
      .first;
}

inline bool mask_support_meets_tile(const Mask<Coord2D> &mask,
                                    const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, tile))
    return true;

  return convex_sets_intersect(
      [&mask](const Coord2D &u) { return mask_support(mask, u); },
      [&tile](const Coord2D &u) { return tile.support(u); },
      mask.bounding_disk_.origin_ - tile.c_radius_.origin_,
      exact_filter_margin(mask.bounding_disk_, tile.c_radius_));
}

// Blueprint placed at the node coord: the tile point closest to the disk center
// lies within the radius
inline bool coord_tile_in_circular_blueprint(const Mask<Coord2D> &mask,
                                             const Coord2D &coord,
                                             const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, coord, tile))
    return true;

  const Coord2D center = coord + mask.bounding_disk_.origin_;
  return leq_test(distance2(center, tile.project_point_to_surface(center)),
                  mask.bounding_disk_.radius2_);
}

// The disk test is cheaper than the query and exact for a circle, so other
// shapes first reject tiles out of reach of the disk placed at the node
inline bool coord_tile_in_blueprint_support(const Mask<Coord2D> &mask,
                                            const Coord2D &coord,
                                            const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, coord, tile))
    return true;

  const CircumscribedRadius<Coord2D> disk{
      .radius2_ = mask.bounding_disk_.radius2_,
      .origin_ = coord + mask.bounding_disk_.origin_};
  if (!disk.overlapping_radi(tile.c_radius_).first)
    return false;

  return convex_sets_intersect(
      [&coord, &mask](const Coord2D &u) {
        return coord + mask_support(mask, u);
      },
      [&tile](const Coord2D &u) { return tile.support(u); },
      coord + mask.bounding_disk_.origin_ - tile.c_radius_.origin_,
      exact_filter_margin(
          mask.bounding_disk_,
          CircumscribedRadius<Coord2D>{.radius2_ = 0., .origin_ = coord},
          tile.c_radius_));
}

// Blueprint tile pairs: A' = A + disk center and B lie within the radius of
// each other
inline bool tile_pair_in_circular_blueprint(const Mask<Coord2D> &mask,
                                            const Tile<Coord2D> &a,
                                            const Tile<Coord2D> &b) {
  return tiles_within_range_2D(a, b, mask.bounding_disk_.origin_,
                               mask.bounding_disk_.radius2_);
}

// The blueprint meets the displacements B - A, whose support is
// b.support(u) - a.support(-u)
inline bool tile_pair_in_blueprint_support(const Mask<Coord2D> &mask,
                                           const Tile<Coord2D> &a,
                                           const Tile<Coord2D> &b) {
  return convex_sets_intersect(
      [&mask](const Coord2D &u) { return mask_support(mask, u); },
      [&a, &b](const Coord2D &u) {
        return b.support(u) - a.support(u.negative());
      },
      mask.bounding_disk_.origin_ - (b.c_radius_.origin_ - a.c_radius_.origin_),
      exact_filter_margin(mask.bounding_disk_, a.c_radius_, b.c_radius_));
}

// Annulus tile tests. The ring is not convex and its anchor (the center) is
// not in it, so it has none of the convex shortcuts. A tile (or, for blueprint
// pairs, the displacement set B - A - anchor) is convex and connected: it
// meets the closed ring iff it is not inside the open inner disk and meets the
// outer disk (it then holds points at distance <= r_out and >= r_in, hence one
// in between). It is inside the inner disk iff its farthest vertex is, so that
// reject is exact; it applies only below r_in - margin, which keeps it sound.
// After it, the outer disk decides through the circle's tests. Those start
// with the anchor (here the center) in the tile, which the perimeter distance
// would miss and which, after the reject, means the tile meets the ring.

// Whether a convex set whose farthest vertex lies farthest_distance2 away from
// the center is inside the inner disk, by more than margin
inline bool inside_annulus_hole(const Mask<Coord2D> &mask,
                                const space_t farthest_distance2,
                                const space_t margin) {
  const space_t reach = std::sqrt(mask.helper_scalars_[0]) - margin;
  return 0. < reach && farthest_distance2 < reach * reach;
}

// FULL only if the tile also misses the open inner disk (by more than the
// margin): its point closest to the center lies at r_in or beyond. With
// r_in = 0 that disk is empty and the annulus is the closed disk, which is
// convex.
inline OVERLAP_LEVEL
overlap_with_all_vertices_in_annular_mask(const Mask<Coord2D> &mask,
                                          const Tile<Coord2D> &tile) {
  if (!(0. < mask.helper_scalars_[0]))
    return OVERLAP_LEVEL::FULL;

  if (tile.coord_in_tile(mask.bounding_disk_.origin_))
    return OVERLAP_LEVEL::PARTIAL;

  const space_t reach =
      std::sqrt(mask.helper_scalars_[0]) +
      exact_filter_margin(mask.bounding_disk_, tile.c_radius_);
  return leq_test(reach * reach, distance2(mask.bounding_disk_.origin_,
                                           tile.project_point_to_surface(
                                               mask.bounding_disk_.origin_)))
             ? OVERLAP_LEVEL::FULL
             : OVERLAP_LEVEL::PARTIAL;
}

inline bool annular_mask_meets_tile(const Mask<Coord2D> &mask,
                                    const Tile<Coord2D> &tile) {
  if (inside_annulus_hole(
          mask, max_distance2(tile.vertices_, mask.bounding_disk_.origin_),
          exact_filter_margin(mask.bounding_disk_, tile.c_radius_)))
    return false;

  return circular_mask_meets_tile(mask, tile);
}

// Blueprint placed at the node coord
inline bool coord_tile_in_annular_blueprint(const Mask<Coord2D> &mask,
                                            const Coord2D &coord,
                                            const Tile<Coord2D> &tile) {
  const Coord2D center = coord + mask.bounding_disk_.origin_;
  if (inside_annulus_hole(
          mask, max_distance2(tile.vertices_, center),
          exact_filter_margin(
              mask.bounding_disk_,
              CircumscribedRadius<Coord2D>{.radius2_ = 0., .origin_ = coord},
              tile.c_radius_)))
    return false;

  return coord_tile_in_circular_blueprint(mask, coord, tile);
}

// Blueprint tile pairs: the displacements B - A - anchor
inline bool tile_pair_in_annular_blueprint(const Mask<Coord2D> &mask,
                                           const Tile<Coord2D> &a,
                                           const Tile<Coord2D> &b) {
  if (inside_annulus_hole(
          mask,
          max_difference_norm2(a.vertices_, b.vertices_,
                               mask.bounding_disk_.origin_),
          exact_filter_margin(mask.bounding_disk_, a.c_radius_, b.c_radius_)))
    return false;

  return tile_pair_in_circular_blueprint(mask, a, b);
}
} // namespace sapi

#endif
