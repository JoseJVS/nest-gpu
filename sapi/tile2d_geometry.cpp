/*
 *  tile2d_geometry.cpp
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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "coordinate_geometry.h"
#include "tile.h"
#include "tile2d_geometry.h"
#include "type_erasure_helpers.h"

namespace sapi {
void initialize_vertices_2D(std::vector<Coord2D> &vertices,
                            CircumscribedRadius<Coord2D> &c_radius,
                            const std::vector<space_t> &side_lengths,
                            const angle_t angular_offset,
                            const bool triangular_vertices) {
  assert(vertices.empty());
  // Side values and radius are checked in validate_tile_parameters_2D
  for ([[maybe_unused]] const auto &side : side_lengths)
    assert(!almost_zero(side * side) && !std::signbit(side));

  const auto half_width = side_lengths[0] / 2;
  const auto half_height =
      1 < side_lengths.size() ? side_lengths[1] / 2 : half_width;

  const vertidx_t total_vertices = triangular_vertices ? 3 : 4;
  vertices.reserve(total_vertices);

  if (angular_offset == 0) {
    for (vertidx_t vert = 0; vert < total_vertices; ++vert) {
      const bool sign_width = vert == 1 || vert == 2;
      const bool sign_height = 1 < vert;

      vertices.emplace_back(construct_coord_2D(
          c_radius.origin_.x_ + (sign_width ? -half_width : half_width),
          c_radius.origin_.y_ + (sign_height ? -half_height : half_height)));
    }
  } else {
    const auto rotation = create_angular_offset(angular_offset);
    for (vertidx_t vert = 0; vert < total_vertices; ++vert) {
      const bool sign_width = vert == 1 || vert == 2;
      const bool sign_height = 1 < vert;

      vertices.emplace_back(
          c_radius.origin_ +
          rotate_displacement(
              construct_coord_2D(sign_width ? -half_width : half_width,
                                 sign_height ? -half_height : half_height),
              rotation));
    }
  }

  c_radius.radius2_ = distance2(c_radius.origin_, vertices[0]);
  assert(!almost_zero(c_radius.radius2_));
}

void initialize_hexagon_vertices(std::vector<Coord2D> &vertices,
                                 CircumscribedRadius<Coord2D> &c_radius,
                                 const space_t side_length,
                                 const angle_t angular_offset) {
  assert(vertices.empty());

  c_radius.radius2_ = side_length * side_length;
  // Checked in validate_tile_parameters_2D
  assert(!std::signbit(side_length) && !almost_zero(c_radius.radius2_));

  vertices.reserve(6);
  for (angle_t angle_ix = 0; angle_ix < 6; ++angle_ix)
    vertices.emplace_back(
        c_radius.origin_ +
        (create_angular_offset(angle_ix * 60 + angular_offset) * side_length));
}

void initialize_helpers_2D(std::vector<Coord2D> &helper_vectors,
                           std::vector<space_t> &helper_scalars,
                           const std::vector<Coord2D> &vertices) {
  assert((vertices.size() == 3 || vertices.size() == 4) &&
         helper_vectors.empty() && helper_scalars.empty());

  helper_vectors.resize(2);
  helper_vectors[0] = vertices[0] - vertices[1];
  helper_vectors[1] = vertices[2] - vertices[1];

  helper_scalars.resize(1, cross_2D(helper_vectors[0], helper_vectors[1]));

  assert(!almost_zero(helper_scalars[0]));
}

void initialize_hexagon_helpers(std::vector<Coord2D> &helper_vectors,
                                std::vector<space_t> &helper_scalars,
                                const std::vector<Coord2D> &vertices) {
  assert(vertices.size() == 6 && helper_vectors.empty() &&
         helper_scalars.empty());

  // Hexagon is divided into rectangles composed of opposing edges
  // used for parallelogram projection comparison
  helper_vectors.reserve(6);
  for (vertidx_t vix = 0; vix < 3; ++vix) {
    helper_vectors.emplace_back(vertices[static_cast<std::size_t>(vix + 1)] -
                                vertices[vix]);
    helper_vectors.emplace_back(
        vertices[static_cast<std::size_t>((vix + 4) % 6)] - vertices[vix]);
  }

  helper_scalars.resize(1, cross_2D(helper_vectors[0], helper_vectors[1]));

  assert(!almost_zero(helper_scalars[0]));
}

namespace {
// Shared body of the split taskloops: initializes the helpers of a sub-tile
// and, at the last split, assigns its leaf index and registers it. Returns
// whether the sub-tile has to be split further.
bool prepare_sub_tile(Tile<Coord2D> &sub_tile,
                      std::vector<const Tile<Coord2D> *> &leaf_tiles,
                      const tileidx_t child_prefix, const split_t rem_splits,
                      const bool generate_total_leaves_vector) {
  initialize_helpers_2D(sub_tile.helper_vectors_, sub_tile.helper_scalars_,
                        sub_tile.vertices_);

  if (0 < rem_splits)
    return true;

  sub_tile.index_ = child_prefix;

  if (generate_total_leaves_vector) {
    const auto lt_it = leaf_tiles.begin() + sub_tile.index_;
    assert(lt_it != leaf_tiles.end() && *lt_it == nullptr);
    *lt_it = &sub_tile;
  }

  return false;
}
} // namespace

// Leaf indices are the mixed-radix number of the partitions taken from the
// root, most significant first, with each split's fan-out as its radix (see
// get_possible_sub_tile_branches). prefix carries the digits above this split.
void split_rectangle(std::vector<Tile<Coord2D>> &sub_tiles,
                     std::vector<const Tile<Coord2D> *> &leaf_tiles,
                     const std::vector<Coord2D> &vertices,
                     const CircumscribedRadius<Coord2D> &c_radius,
                     const tileidx_t prefix, const split_t splits,
                     const bool generate_total_leaves_vector) {
  assert(0 < splits && sub_tiles.empty() && 0 <= prefix &&
         leaf_tiles.empty() != generate_total_leaves_vector);

  constexpr split_t branches = split_t(TILE_SPLITS::RECTANGLE_SPLITS);
  sub_tiles.reserve(branches);
  const split_t rem_splits = splits - 1;
  const space_t radius2 = c_radius.radius2_ / 4;
  assert(!almost_zero(radius2));

  Coord2D midpoints[4];
  for (vertidx_t vix = 0; vix < 4; ++vix)
    midpoints[vix] = midpoint(
        vertices[vix], vertices[static_cast<std::size_t>((vix + 1) % 4)]);

  for (split_t partition = 0; partition < branches; ++partition) {
    sub_tiles.emplace_back(Tile<Coord2D>(
        TILE_SHAPE::RECTANGLE,
        construct_circumscribed_radius(
            midpoint(c_radius.origin_, vertices[partition]), radius2),
        {vertices[partition], midpoints[partition], c_radius.origin_,
         midpoints[(partition + 3) % 4]}));
  }

#pragma omp taskloop num_tasks(strict : 4)                                     \
    mergeable final(rem_splits < 3) default(none)                              \
    shared(sub_tiles, leaf_tiles)                                              \
    firstprivate(prefix, rem_splits, generate_total_leaves_vector)
  for (split_t partition = 0; partition < branches; ++partition) {
    auto &sub_tile = sub_tiles[static_cast<std::size_t>(partition)];
    const tileidx_t child_prefix = prefix * branches + partition;

    if (prepare_sub_tile(sub_tile, leaf_tiles, child_prefix, rem_splits,
                         generate_total_leaves_vector))
      split_rectangle(sub_tile.sub_tiles_, leaf_tiles, sub_tile.vertices_,
                      sub_tile.c_radius_, child_prefix, rem_splits,
                      generate_total_leaves_vector);
  }
}

void split_triangle(std::vector<Tile<Coord2D>> &sub_tiles,
                    std::vector<const Tile<Coord2D> *> &leaf_tiles,
                    const std::vector<Coord2D> &vertices,
                    const tileidx_t prefix, const split_t splits,
                    const bool generate_total_leaves_vector) {
  assert(0 < splits && sub_tiles.empty() && 0 <= prefix &&
         leaf_tiles.empty() != generate_total_leaves_vector);

  // The two halves are built explicitly below.
  constexpr split_t branches = split_t(TILE_SPLITS::TRIANGLE_SPLITS);
  static_assert(branches == 2);
  sub_tiles.reserve(branches);
  const split_t rem_splits = splits - 1;

  // Get AB, the largest edge of triangle ABC
  const auto sorted_coords = std::get<1>(sort_triangular_vertices(vertices));

  // Get D midpoint of AB,
  // this midpoint is now a vertex of each sub-triangle DCA and DCB,
  // this guarantees each sub-triangle has same area size.
  const Coord2D coordD(midpoint(sorted_coords[0], sorted_coords[1]));

  std::vector<Coord2D> left = {sorted_coords[2], coordD, sorted_coords[0]};
  sub_tiles.emplace_back(Tile<Coord2D>(
      TILE_SHAPE::TRIANGLE, compute_c_radius(left), std::move(left)));

  std::vector<Coord2D> right = {sorted_coords[2], coordD, sorted_coords[1]};
  sub_tiles.emplace_back(Tile<Coord2D>(
      TILE_SHAPE::TRIANGLE, compute_c_radius(right), std::move(right)));

#pragma omp taskloop num_tasks(strict : 2)                                     \
    mergeable final(rem_splits < 5) default(none)                              \
    shared(sub_tiles, leaf_tiles)                                              \
    firstprivate(prefix, rem_splits, generate_total_leaves_vector)
  for (split_t partition = 0; partition < branches; ++partition) {
    auto &sub_tile = sub_tiles[static_cast<std::size_t>(partition)];
    const tileidx_t child_prefix = prefix * branches + partition;

    if (prepare_sub_tile(sub_tile, leaf_tiles, child_prefix, rem_splits,
                         generate_total_leaves_vector))
      split_triangle(sub_tile.sub_tiles_, leaf_tiles, sub_tile.vertices_,
                     child_prefix, rem_splits, generate_total_leaves_vector);
  }
}

void split_hexagon(std::vector<Tile<Coord2D>> &sub_tiles,
                   std::vector<const Tile<Coord2D> *> &leaf_tiles,
                   const std::vector<Coord2D> &vertices,
                   const CircumscribedRadius<Coord2D> &c_radius,
                   const tileidx_t prefix, const split_t splits,
                   const bool generate_total_leaves_vector) {
  assert(0 < splits && sub_tiles.empty() && 0 <= prefix &&
         leaf_tiles.empty() != generate_total_leaves_vector);

  constexpr split_t branches = split_t(TILE_SPLITS::HEXAGON_SPLITS);
  sub_tiles.reserve(branches);
  const split_t rem_splits = splits - 1;
  const space_t radius2 = c_radius.radius2_ / 3;
  assert(!almost_zero(radius2));

  for (split_t partition = 0; partition < branches; ++partition) {
    const split_t next_partition = static_cast<split_t>((partition + 1) % 6);

    sub_tiles.emplace_back(Tile<Coord2D>(
        TILE_SHAPE::TRIANGLE,
        construct_circumscribed_radius(
            (c_radius.origin_ / 3) +
                ((vertices[partition] + vertices[next_partition]) / 3),
            radius2),
        {c_radius.origin_, vertices[partition], vertices[next_partition]}));
  }

#pragma omp taskloop num_tasks(strict : 3)                                     \
    mergeable final(rem_splits < 4) default(none)                              \
    shared(sub_tiles, leaf_tiles)                                              \
    firstprivate(prefix, rem_splits, generate_total_leaves_vector)
  for (split_t partition = 0; partition < branches; ++partition) {
    auto &sub_tile = sub_tiles[static_cast<std::size_t>(partition)];
    const tileidx_t child_prefix = prefix * branches + partition;

    if (prepare_sub_tile(sub_tile, leaf_tiles, child_prefix, rem_splits,
                         generate_total_leaves_vector))
      split_triangle(sub_tile.sub_tiles_, leaf_tiles, sub_tile.vertices_,
                     child_prefix, rem_splits, generate_total_leaves_vector);
  }
}

bool coord_in_rectangle(const Coord2D &coord,
                        const CircumscribedRadius<Coord2D> &c_radius,
                        const std::vector<Coord2D> &vertices,
                        const std::vector<Coord2D> &helper_vectors,
                        const std::vector<space_t> &helper_scalars) {
  if (!c_radius.coord_in_radius(coord).first)
    return false;

  // Only works for squares or triangles
  return algebraic_projection_comparison<Coord2D, false>(
      coord - vertices[1], helper_vectors[0], helper_vectors[1],
      helper_scalars[0]);
}

bool coord_in_triangle(const Coord2D &coord,
                       const CircumscribedRadius<Coord2D> &c_radius,
                       const std::vector<Coord2D> &vertices,
                       const std::vector<Coord2D> &helper_vectors,
                       const std::vector<space_t> &helper_scalars) {
  if (!c_radius.coord_in_radius(coord).first)
    return false;

  // Only works for squares or triangles
  return algebraic_projection_comparison<Coord2D, true>(
      coord - vertices[1], helper_vectors[0], helper_vectors[1],
      helper_scalars[0]);
}

bool coord_in_hexagon(const Coord2D &coord,
                      const CircumscribedRadius<Coord2D> &c_radius,
                      const std::vector<Coord2D> &vertices,
                      const std::vector<Coord2D> &helper_vectors,
                      const std::vector<space_t> &helper_scalars) {
  if (!c_radius.coord_in_radius(coord).first)
    return false;

  // Parallelogram based comparison using rotating rectangles
  bool res = false;
  vertidx_t rel_vix;
  for (vertidx_t vix = 0; vix < 3; ++vix) {
    rel_vix = 2 * vix;
    res = algebraic_projection_comparison<Coord2D, false>(
        coord - vertices[vix], helper_vectors[rel_vix],
        helper_vectors[rel_vix + 1], helper_scalars[0]);
    if (res)
      break;
  }

  return res;
}

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_rectangle(const nodeidx_t first_index,
                             const nodeidx_t coord_count, AnyRNG &rng,
                             const std::vector<Coord2D> &vertices,
                             const std::vector<Coord2D> &helper_vectors) {
  assert(0 <= first_index && 0 < coord_count);
  std::vector<std::pair<nodeidx_t, Coord2D>> vec;
  vec.reserve(static_cast<std::size_t>(coord_count));

  space_t m0, m1;
  std::uniform_real_distribution<space_t> dist(0., 1.);
  for (nodeidx_t idx = 0; idx < coord_count; ++idx) {
    m0 = static_cast<space_t>(dist(rng));
    m1 = static_cast<space_t>(dist(rng));
    vec.emplace_back(std::make_pair(
        first_index + idx,
        construct_coord_2D(vertices[1].x_ + helper_vectors[0].x_ * m0 +
                               helper_vectors[1].x_ * m1,
                           vertices[1].y_ + helper_vectors[0].y_ * m0 +
                               helper_vectors[1].y_ * m1)));
  }

  return vec;
}

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_triangle(const nodeidx_t first_index,
                            const nodeidx_t coord_count, AnyRNG &rng,
                            const std::vector<Coord2D> &vertices) {
  assert(0 <= first_index && 0 < coord_count);
  std::vector<std::pair<nodeidx_t, Coord2D>> vec;
  vec.reserve(static_cast<std::size_t>(coord_count));

  space_t m0, m1, m2;
  std::uniform_real_distribution<space_t> dist(0., 1.);
  for (nodeidx_t idx = 0; idx < coord_count; ++idx) {
    m0 = static_cast<space_t>(dist(rng));
    m1 = static_cast<space_t>(dist(rng));
    const bool adjust = std::isless(1., m0 + m1);
    m0 = adjust * std::fmax(std::fmin(1., 1. - m0), 0.) + !adjust * m0;
    m1 = adjust * std::fmax(std::fmin(1., 1. - m1), 0.) + !adjust * m1;
    m2 = std::fmax(std::fmin(1., 1. - m0 - m1), 0.);
    vec.emplace_back(std::make_pair(
        first_index + idx,
        construct_coord_2D(
            vertices[0].x_ * m0 + vertices[1].x_ * m1 + vertices[2].x_ * m2,
            vertices[0].y_ * m0 + vertices[1].y_ * m1 + vertices[2].y_ * m2)));
  }

  return vec;
}

std::vector<std::pair<nodeidx_t, Coord2D>>
generate_coords_in_hexagon(const nodeidx_t first_index,
                           const nodeidx_t coord_count, AnyRNG &rng,
                           const std::vector<Coord2D> &vertices,
                           const CircumscribedRadius<Coord2D> &c_radius) {
  assert(0 <= first_index && 0 < coord_count);
  std::vector<std::pair<nodeidx_t, Coord2D>> vec;
  vec.reserve(static_cast<std::size_t>(coord_count));

  vertidx_t hex_idx;
  space_t m0, m1, m2;
  std::uniform_real_distribution<space_t> space_dist(0., 1.);
  std::uniform_int_distribution<uniform_int_type_t<vertidx_t>> partition_dist(
      0, 5);
  for (nodeidx_t idx = 0; idx < coord_count; ++idx) {
    m0 = static_cast<space_t>(space_dist(rng));
    m1 = static_cast<space_t>(space_dist(rng));
    const bool adjust = std::isless(1., m0 + m1);
    m0 = adjust * std::fmax(std::fmin(1., 1. - m0), 0.) + !adjust * m0;
    m1 = adjust * std::fmax(std::fmin(1., 1. - m1), 0.) + !adjust * m1;
    m2 = std::fmax(std::fmin(1., 1. - m0 - m1), 0.);
    hex_idx = static_cast<vertidx_t>(partition_dist(rng));
    vec.emplace_back(std::make_pair(
        first_index + idx,
        construct_coord_2D(
            c_radius.origin_.x_ * m0 + vertices[hex_idx].x_ * m1 +
                vertices[static_cast<std::size_t>((hex_idx + 1) % 6)].x_ * m2,
            c_radius.origin_.y_ * m0 + vertices[hex_idx].y_ * m1 +
                vertices[static_cast<std::size_t>((hex_idx + 1) % 6)].y_ *
                    m2)));
  }

  return vec;
}

// Exact for convex polygons with vertices in perimeter order: every edge is
// projected and the closest result is kept. Each edge is projected from its
// nearer endpoint (the lower index on ties).
Coord2D
project_point_to_polygon_perimeter(const Coord2D &coord,
                                   const std::vector<Coord2D> &vertices) {
  // 2D tiles have at most 6 vertices (hexagons)
  std::array<space_t, 6> vertex_d2;
  assert(2 < vertices.size() && vertices.size() <= vertex_d2.size());
  const vertidx_t num_vertices = static_cast<vertidx_t>(vertices.size());

  for (vertidx_t index = 0; index < num_vertices; ++index) {
    vertex_d2[index] = distance2(coord, vertices[index]);
    if (almost_zero(vertex_d2[index]))
      return vertices[index];
  }

  Coord2D closest = vertices[0];
  space_t closest_d2 = std::numeric_limits<space_t>::max();
  for (vertidx_t first = 0; first < num_vertices; ++first) {
    const vertidx_t second = static_cast<vertidx_t>((first + 1) % num_vertices);
    const bool second_is_base =
        std::isless(vertex_d2[second], vertex_d2[first]) ||
        (vertex_d2[second] == vertex_d2[first] && second < first);
    const auto projected = projection_coord<Coord2D, true>(
        coord, vertices[second_is_base ? second : first],
        vertices[second_is_base ? first : second]);
    const auto d2 = distance2(coord, projected);
    if (std::isless(d2, closest_d2)) {
      closest_d2 = d2;
      closest = projected;
    }
  }

  return closest;
}

Coord2D project_point_to_triangle_perimeter(
    const Coord2D &coord, const std::vector<Coord2D> &vertices,
    const CircumscribedRadius<Coord2D> &c_radius) {
  if (almost_zero(distance2(coord, c_radius.origin_)))
    return c_radius.origin_;

  return project_point_to_polygon_perimeter(coord, vertices);
}

void initialize_tile_vertices_2D(std::vector<Coord2D> &vertices,
                                 CircumscribedRadius<Coord2D> &c_radius,
                                 const std::vector<space_t> &side_lengths,
                                 const std::vector<angle_t> &angular_offsets,
                                 const TILE_SHAPE shape) {
  // Runs per tile inside grid generation tasks: the parameters were checked
  // once by initialize_ctc (validate_tile_parameters_2D) before the region
  assert(angular_offsets.size() < 2);
  const angle_t angular_offset =
      angular_offsets.empty() ? angle_t(0) : angular_offsets[0];

  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    initialize_vertices_2D(vertices, c_radius, side_lengths, angular_offset,
                           false // triangular vertices
    );
    break;

  case TILE_SHAPE::TRIANGLE:
    initialize_vertices_2D(vertices, c_radius, side_lengths, angular_offset,
                           true // triangular vertices
    );
    break;

  case TILE_SHAPE::HEXAGON:
    initialize_hexagon_vertices(vertices, c_radius, side_lengths[0],
                                angular_offset);
    break;

  default:
    assert(false && "Invalid tile shape");
  }
}

tileidx_t compute_leaves_count_2D(const split_t splits,
                                  const TILE_SHAPE shape) {
  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    return static_cast<tileidx_t>(
        std::pow(uint8_t(TILE_SPLITS::RECTANGLE_SPLITS), splits));

  case TILE_SHAPE::TRIANGLE:
    return static_cast<tileidx_t>(
        std::pow(uint8_t(TILE_SPLITS::TRIANGLE_SPLITS), splits));

  case TILE_SHAPE::HEXAGON:
    return static_cast<tileidx_t>(
        splits > 0
            ? uint8_t(TILE_SPLITS::HEXAGON_SPLITS) *
                  std::pow(uint8_t(TILE_SPLITS::TRIANGLE_SPLITS), splits - 1)
            : 1);

  default:
    assert(false && "Invalid tile shape");
    return 1;
  }
}

std::vector<split_t> get_possible_sub_tile_branches_2D(const split_t splits,
                                                       const TILE_SHAPE shape) {
  if (splits == 0)
    return {};

  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    return std::vector<split_t>(splits, split_t(TILE_SPLITS::RECTANGLE_SPLITS));

  case TILE_SHAPE::TRIANGLE:
    return std::vector<split_t>(splits, split_t(TILE_SPLITS::TRIANGLE_SPLITS));

  case TILE_SHAPE::HEXAGON: {
    std::vector<split_t> split_vec(splits,
                                   split_t(TILE_SPLITS::TRIANGLE_SPLITS));
    split_vec[0] = split_t(TILE_SPLITS::HEXAGON_SPLITS);
    return split_vec;
  }

  default:
    assert(false && "Invalid tile shape");
    return {};
  }
}

split_t compute_minimal_splits_2D(const double expected_total_nodes,
                                  const double total_tiles,
                                  const double expected_nodes_per_leaf,
                                  const TILE_SHAPE shape) {
  switch (shape) {
  case TILE_SHAPE::RECTANGLE:
    return split_t(
        std::max(std::floor(std::log(expected_total_nodes /
                                     (expected_nodes_per_leaf * total_tiles)) /
                            std::log(4.)),
                 0.));

  case TILE_SHAPE::TRIANGLE:
    return split_t(
        std::max(std::floor(std::log(expected_total_nodes /
                                     (expected_nodes_per_leaf * total_tiles)) /
                            std::log(2.)),
                 0.));

  case TILE_SHAPE::HEXAGON:
    return split_t(std::max(
        std::floor(std::log(expected_total_nodes /
                            (6. * expected_nodes_per_leaf * total_tiles)) /
                       std::log(2.) +
                   1.),
        0.));

  default:
    throw std::invalid_argument("Invalid tile shape");
  }
}

// Stages after the circumscribed-circle check of Mask::tiles_within_mask_range:
// the polygon distance between A' = A + offset and B, then overlap of A' and B
// without a vertex close to the other perimeter
bool tiles_within_range_2D(const Tile<Coord2D> &a, const Tile<Coord2D> &b,
                           const Coord2D &offset, const space_t radius2) {
  if (leq_test(std::fmin(min_vertex_edge_distance2_2D(a.vertices_, offset,
                                                      b.vertices_),
                         min_vertex_edge_distance2_2D(
                             b.vertices_, offset.negative(), a.vertices_)),
               radius2))
    return true;

  return std::ranges::any_of(
             a.vertices_,
             [&](const Coord2D &v) { return b.coord_in_tile(v + offset); }) ||
         std::ranges::any_of(
             b.vertices_,
             [&](const Coord2D &w) { return a.coord_in_tile(w - offset); }) ||
         edges_cross_2D(a.vertices_, offset, b.vertices_);
}

// Minimum squared distance from the vertices of `from`, translated by `shift`,
// to the perimeter of `to`. For disjoint convex polygons the closest pair of
// points always includes a vertex, so the minimum over both directions is the
// polygon distance.
space_t min_vertex_edge_distance2_2D(const std::vector<Coord2D> &from,
                                     const Coord2D &shift,
                                     const std::vector<Coord2D> &to) {
  space_t min_d2 = std::numeric_limits<space_t>::max();
  for (const auto &vertex : from) {
    const Coord2D shifted = vertex + shift;
    min_d2 = std::fmin(
        min_d2,
        distance2(shifted, project_point_to_polygon_perimeter(shifted, to)));
  }
  return min_d2;
}

// Whether an edge of `a` translated by `shift` properly crosses an edge of `b`
// (strict orientation signs: touching contacts are not crossings)
bool edges_cross_2D(const std::vector<Coord2D> &a, const Coord2D &shift,
                    const std::vector<Coord2D> &b) {
  const vertidx_t a_count = static_cast<vertidx_t>(a.size());
  const vertidx_t b_count = static_cast<vertidx_t>(b.size());
  for (vertidx_t i = 0; i < a_count; ++i) {
    const Coord2D a0 = a[i] + shift;
    const Coord2D a1 = a[static_cast<vertidx_t>((i + 1) % a_count)] + shift;
    for (vertidx_t j = 0; j < b_count; ++j) {
      const Coord2D &b0 = b[j];
      const Coord2D &b1 = b[static_cast<vertidx_t>((j + 1) % b_count)];
      if (orientation(a0, a1, b0) * orientation(a0, a1, b1) < 0 &&
          orientation(b0, b1, a0) * orientation(b0, b1, a1) < 0)
        return true;
    }
  }
  return false;
}
} // namespace sapi
