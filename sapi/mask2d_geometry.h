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

// Centred shapes (circle, ellipse, parallelogram): the bounding disc is centred
// on the anchor, so the disc displacement is the anchor displacement
inline OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &coord,
                                               const Mask<Coord2D> &mask) {
  return mask.bounding_disc_.coord_in_radius(coord);
}

inline OptDisp<Coord2D> coord_in_circular_mask(const Coord2D &a,
                                               const Coord2D &b,
                                               const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  od.first = leq_test(od.second.distance2_, mask.bounding_disc_.radius2_);
  return od;
}

inline OptDisp<Coord2D> coord_in_elliptical_mask(const Coord2D &coord,
                                                 const Mask<Coord2D> &mask) {
  auto od = mask.bounding_disc_.coord_in_radius(coord);
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
  if (leq_test(od.second.distance2_, mask.bounding_disc_.radius2_))
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
  auto od = mask.bounding_disc_.coord_in_radius(coord);
  if (od.first)
    od.first = algebraic_projection_comparison<Coord2D, false>(
        od.second.displacement_ + mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_vectors_[2],
        mask.helper_scalars_[0]);

  return od;
}

inline OptDisp<Coord2D> coord_in_parallelogram_mask(const Coord2D &a,
                                                    const Coord2D &b,
                                                    const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  if (leq_test(od.second.distance2_, mask.bounding_disc_.radius2_))
    od.first = algebraic_projection_comparison<Coord2D, false>(
        od.second.displacement_ + mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_vectors_[2],
        mask.helper_scalars_[0]);

  return od;
}

// The triangle is defined relative to its anchor vertex while its bounding
// disc is centred on the triangle: the prefilter uses the disc, the exact test
// and the returned displacement use the anchor
inline OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &coord,
                                                 const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(coord - mask.anchor_));
  if (mask.bounding_disc_.coord_in_radius(coord).first)
    od.first = algebraic_projection_comparison<Coord2D, true>(
        od.second.displacement_, mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_scalars_[0]);

  return od;
}

inline OptDisp<Coord2D> coord_in_triangular_mask(const Coord2D &a,
                                                 const Coord2D &b,
                                                 const Mask<Coord2D> &mask) {
  OptDisp<Coord2D> od(false, construct_displacement(b - (a + mask.anchor_)));
  if (leq_test(vector_norm2(b - (a + mask.bounding_disc_.origin_)),
               mask.bounding_disc_.radius2_))
    od.first = algebraic_projection_comparison<Coord2D, true>(
        od.second.displacement_, mask.helper_vectors_[0],
        mask.helper_vectors_[1], mask.helper_scalars_[0]);

  return od;
}

// Named views of the helper layout set by the initializers in
// mask2d_geometry.cpp, relative to the anchor.
// Ellipse: helper_vectors_[0] holds the squared semi-axes (a^2, b^2) and, when
// rotated, helper_vectors_[1] the (cos, sin) of the rotation.
inline const Coord2D &ellipse_squared_semi_axes(const Mask<Coord2D> &mask) {
  return mask.helper_vectors_[0];
}

// Parallelogram: helper_vectors_ holds a0, a0 - a1, a0 + a1 and a1, for the
// vertices a0, a1, -a0, -a1.
inline std::array<Coord2D, 4>
parallelogram_vertices(const Mask<Coord2D> &mask) {
  const Coord2D &a0 = mask.helper_vectors_[0];
  const Coord2D &a1 = mask.helper_vectors_[3];
  return {a0, a1, a0.negative(), a1.negative()};
}

// Triangle: helper_vectors_ holds a0 and a1, for the vertices 0 (the anchor),
// a0 and a1.
inline std::array<Coord2D, 3> triangle_vertices(const Mask<Coord2D> &mask) {
  return {Coord2D(), mask.helper_vectors_[0], mask.helper_vectors_[1]};
}

// Support point of the ellipse centred at the origin: in the frame of its
// axes, (a^2 q_x, b^2 q_y) / sqrt(a^2 q_x^2 + b^2 q_y^2) for the direction q
// (the unit-circle support under the affine map), rotated back. The centre
// for a zero direction.
inline Coord2D ellipse_support(const Mask<Coord2D> &mask,
                               const Coord2D &direction) {
  const Coord2D &axes2 = ellipse_squared_semi_axes(mask);
  const bool rotated = 1 < mask.helper_vectors_.size();
  const Coord2D q =
      rotated ? rotate_displacement(direction, mask.helper_vectors_[1])
              : direction;
  const space_t norm = std::sqrt((axes2 * q * q).sum());
  if (!(0 < norm))
    return Coord2D();

  const Coord2D point = axes2 * q / norm;
  if (!rotated)
    return point;
  const Coord2D &rotation = mask.helper_vectors_[1];
  return rotate_displacement(point,
                             construct_coord_2D(rotation.x_, -rotation.y_));
}

// Support point of the mask: its farthest point along direction, in the frame
// of anchor_ (absolute for source/target masks, an offset from the driver node
// for blueprints)
inline Coord2D mask_support(const Mask<Coord2D> &mask,
                            const Coord2D &direction) {
  switch (mask.shape_) {
  case MASK_SHAPE::CIRCULAR: {
    // No absolute threshold on the direction: GJK directions shrink towards
    // zero near touching
    const space_t norm = vector_norm(direction);
    return 0 < norm ? mask.anchor_ +
                          direction *
                              (std::sqrt(mask.bounding_disc_.radius2_) / norm)
                    : mask.anchor_;
  }
  case MASK_SHAPE::ELLIPTICAL:
    return mask.anchor_ + ellipse_support(mask, direction);
  case MASK_SHAPE::PARALLELOGRAM:
    return mask.anchor_ +
           farthest_point(parallelogram_vertices(mask), direction);
  case MASK_SHAPE::TRIANGULAR:
    return mask.anchor_ + farthest_point(triangle_vertices(mask), direction);
  default:
    assert(false && "Invalid mask shape");
    return mask.anchor_;
  }
}

// Tile tests per shape, dispatched from mask_geometry.h after the generic fast
// tests of Mask. A circle equals its bounding disc, so the disc decides it
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
// mask (the cheap accept the tests below start with)
inline bool anchor_in_tile(const Mask<Coord2D> &mask,
                           const Tile<Coord2D> &tile) {
  return tile.coord_in_tile(mask.anchor_);
}

// Blueprint placed at the node coord
inline bool anchor_in_tile(const Mask<Coord2D> &mask, const Coord2D &coord,
                           const Tile<Coord2D> &tile) {
  return tile.coord_in_tile(coord + mask.anchor_);
}

// Source/target masks: the tile perimeter point closest to the centre lies
// within the radius
inline bool circular_mask_meets_tile(const Mask<Coord2D> &mask,
                                     const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, tile))
    return true;

  return mask.bounding_disc_
      .coord_in_radius(
          tile.project_point_to_surface(mask.bounding_disc_.origin_))
      .first;
}

inline bool mask_support_meets_tile(const Mask<Coord2D> &mask,
                                    const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, tile))
    return true;

  return convex_sets_intersect(
      [&mask](const Coord2D &u) { return mask_support(mask, u); },
      [&tile](const Coord2D &u) { return tile.support(u); },
      mask.bounding_disc_.origin_ - tile.c_radius_.origin_,
      exact_filter_margin(mask.bounding_disc_, tile.c_radius_));
}

// Blueprint placed at the node coord: the tile point closest to the disc centre
// lies within the radius
inline bool coord_tile_in_circular_blueprint(const Mask<Coord2D> &mask,
                                             const Coord2D &coord,
                                             const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, coord, tile))
    return true;

  const Coord2D centre = coord + mask.bounding_disc_.origin_;
  return leq_test(distance2(centre, tile.project_point_to_surface(centre)),
                  mask.bounding_disc_.radius2_);
}

// The disc test is cheaper than the query and exact for a circle, so other
// shapes first reject tiles out of reach of the disc placed at the node
inline bool coord_tile_in_blueprint_support(const Mask<Coord2D> &mask,
                                            const Coord2D &coord,
                                            const Tile<Coord2D> &tile) {
  if (anchor_in_tile(mask, coord, tile))
    return true;

  const CircumscribedRadius<Coord2D> disc{
      .radius2_ = mask.bounding_disc_.radius2_,
      .origin_ = coord + mask.bounding_disc_.origin_};
  if (!disc.overlapping_radi(tile.c_radius_).first)
    return false;

  return convex_sets_intersect(
      [&coord, &mask](const Coord2D &u) {
        return coord + mask_support(mask, u);
      },
      [&tile](const Coord2D &u) { return tile.support(u); },
      coord + mask.bounding_disc_.origin_ - tile.c_radius_.origin_,
      exact_filter_margin(
          mask.bounding_disc_,
          CircumscribedRadius<Coord2D>{.radius2_ = 0., .origin_ = coord},
          tile.c_radius_));
}

// Blueprint tile pairs: A' = A + disc centre and B lie within the radius of
// each other
inline bool tile_pair_in_circular_blueprint(const Mask<Coord2D> &mask,
                                            const Tile<Coord2D> &a,
                                            const Tile<Coord2D> &b) {
  return tiles_within_range_2D(a, b, mask.bounding_disc_.origin_,
                               mask.bounding_disc_.radius2_);
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
      mask.bounding_disc_.origin_ - (b.c_radius_.origin_ - a.c_radius_.origin_),
      exact_filter_margin(mask.bounding_disc_, a.c_radius_, b.c_radius_));
}
} // namespace sapi

#endif
