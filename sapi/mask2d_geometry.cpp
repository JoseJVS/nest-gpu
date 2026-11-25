/*
 *  mask2d_geometry.cpp
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

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

#include "coordinate_geometry.h"
#include "mask2d_geometry.h"

namespace sapi {
void initialize_circular_mask_helpers(Mask<Coord2D> &mask,
                                      const std::vector<space_t> &mask_params) {
  if (mask_params.size() != 1)
    throw std::invalid_argument("Invalid circular mask params vector");

  mask.bounding_disc_.radius2_ = mask_params[0] * mask_params[0];
  if (std::signbit(mask_params[0]) || almost_zero(mask.bounding_disc_.radius2_))
    throw std::invalid_argument("Invalid circular mask param");

  mask.bounding_disc_.origin_ = mask.anchor_;
}

// Closed ring r_in <= |p - anchor| <= r_out, r_in may be 0 (a disc); the
// bounding disc is the outer one
void initialize_annular_mask_helpers(Mask<Coord2D> &mask,
                                     const std::vector<space_t> &mask_params) {
  if (mask_params.size() != 2)
    throw std::invalid_argument("Invalid annular mask params vector");

  const space_t inner_radius = mask_params[0];
  const space_t outer_radius = mask_params[1];
  if (std::signbit(inner_radius) || !(inner_radius < outer_radius) ||
      almost_zero(outer_radius * outer_radius))
    throw std::invalid_argument("Invalid annular mask radii");

  mask.bounding_disc_.radius2_ = outer_radius * outer_radius;
  mask.bounding_disc_.origin_ = mask.anchor_;
  mask.helper_scalars_.resize(1, inner_radius * inner_radius);
}

void initialize_elliptical_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params) {
  if (mask_params.size() < 2 || 3 < mask_params.size())
    throw std::invalid_argument("Invalid elliptical mask params vector");

  if (std::signbit(mask_params[0]) || std::signbit(mask_params[1]))
    throw std::invalid_argument("Invalid elliptical mask axes");

  Coord2D axes = construct_coord_2D(mask_params[0] * mask_params[0],
                                    mask_params[1] * mask_params[1]);

  if (almost_zero(axes.x_) || almost_zero(axes.y_))
    throw std::invalid_argument("Invalid elliptical mask axes");

  mask.bounding_disc_.radius2_ = std::fmax(axes.x_, axes.y_);

  const space_t rotation = 2 < mask_params.size() ? mask_params[2] : 0;
  if (almost_zero(rotation)) {
    mask.helper_vectors_.resize(1, axes);
  } else {
    mask.helper_vectors_.resize(2);
    mask.helper_vectors_[0] = axes;
    mask.helper_vectors_[1] = create_angular_offset(rotation);
  }

  mask.bounding_disc_.origin_ = mask.anchor_;
}

void initialize_parallelogram_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params) {
  if (mask_params.size() != 4)
    throw std::invalid_argument("Invalid parallelogram mask params vector");

  auto it = mask_params.begin();
  auto axial_vector0 = Coord2D::copy_from_vec(it);
  auto axial_vector1 = Coord2D::copy_from_vec(it);

  mask.bounding_disc_.radius2_ =
      std::fmax(vector_norm2(axial_vector0), vector_norm2(axial_vector1));

  if (almost_zero(mask.bounding_disc_.radius2_))
    throw std::invalid_argument("Invalid parallelogram mask vectors");

  mask.helper_vectors_.resize(4);
  mask.helper_vectors_[0] = axial_vector0;
  mask.helper_vectors_[1] = axial_vector0 - axial_vector1;
  mask.helper_vectors_[2] = axial_vector0 + axial_vector1;
  mask.helper_vectors_[3] = axial_vector1;

  mask.helper_scalars_.resize(
      1, cross_2D(mask.helper_vectors_[1], mask.helper_vectors_[2]));

  if (almost_zero(mask.helper_scalars_[0]))
    throw std::invalid_argument("Invalid parallelogram mask vectors");

  mask.bounding_disc_.origin_ = mask.anchor_;
}

namespace {
// Smallest disc containing the triangle {0, a0, a1}: centred at the midpoint
// of the longest edge when the triangle is obtuse (or right), at the
// circumcentre otherwise. Its radius reaches the farthest vertex, widened by
// EXACT_FILTER_REL_TOL so rounding never leaves a vertex outside. Set in the
// frame of the anchor vertex.
void set_triangle_bounding_disc(Mask<Coord2D> &mask, const Coord2D &a0,
                                const Coord2D &a1) {
  const std::array<Coord2D, 3> vertices{Coord2D{}, a0, a1};

  // Vertex with an obtuse (or right) angle, if any
  std::size_t obtuse = vertices.size();
  for (std::size_t vix = 0; vix < vertices.size(); ++vix) {
    const auto &v = vertices[vix];
    const auto &p = vertices[(vix + 1) % vertices.size()];
    const auto &q = vertices[(vix + 2) % vertices.size()];
    if (!std::isgreater(((p - v) * (q - v)).sum(), 0.))
      obtuse = vix;
  }

  Coord2D center;
  if (obtuse < vertices.size())
    center = (vertices[(obtuse + 1) % vertices.size()] +
              vertices[(obtuse + 2) % vertices.size()]) *
             0.5;
  else
    center = circumscribed_center(
        std::vector<Coord2D>(vertices.begin(), vertices.end()));

  space_t radius2 = 0.;
  for (const auto &v : vertices)
    radius2 = std::fmax(radius2, vector_norm2(v - center));

  // The centre lies in the closed triangle: on the same side of every edge as
  // the triangle, up to rounding. The cross products' rounding scales with the
  // squared edge lengths, not with the area, so the tolerance does too (thin
  // slivers are valid masks).
  [[maybe_unused]] const bool positive = 0. < cross_2D(a0, a1);
  [[maybe_unused]] space_t max_edge2 = 0.;
  for (std::size_t vix = 0; vix < vertices.size(); ++vix)
    max_edge2 = std::fmax(
        max_edge2,
        vector_norm2(vertices[(vix + 1) % vertices.size()] - vertices[vix]));
  for (std::size_t vix = 0; vix < vertices.size(); ++vix) {
    [[maybe_unused]] const space_t side =
        cross_2D(vertices[(vix + 1) % vertices.size()] - vertices[vix],
                 center - vertices[vix]);
    assert(-EXACT_FILTER_REL_TOL * max_edge2 <= (positive ? side : -side));
  }

  mask.bounding_disc_.origin_ = mask.anchor_ + center;
  mask.bounding_disc_.radius2_ = radius2 * (1 + 2 * EXACT_FILTER_REL_TOL);
}
} // namespace

void initialize_triangular_mask_helpers(
    Mask<Coord2D> &mask, const std::vector<space_t> &mask_params) {
  if (mask_params.size() != 4)
    throw std::invalid_argument("Invalid triangular mask params vector");

  auto it = mask_params.begin();
  auto axial_vector0 = Coord2D::copy_from_vec(it);
  auto axial_vector1 = Coord2D::copy_from_vec(it);

  mask.bounding_disc_.radius2_ =
      std::fmax(vector_norm2(axial_vector0), vector_norm2(axial_vector1));

  if (almost_zero(mask.bounding_disc_.radius2_))
    throw std::invalid_argument("Invalid triangular mask vectors");

  mask.helper_scalars_.resize(1, cross_2D(axial_vector0, axial_vector1));

  if (almost_zero(mask.helper_scalars_[0]))
    throw std::invalid_argument("Invalid triangular mask vectors");

  mask.helper_vectors_.resize(2);
  mask.helper_vectors_[0] = axial_vector0;
  mask.helper_vectors_[1] = axial_vector1;

  // Bounding disc centred on the triangle {0, a0, a1} (relative to the
  // anchor vertex)
  set_triangle_bounding_disc(mask, axial_vector0, axial_vector1);
}
} // namespace sapi
