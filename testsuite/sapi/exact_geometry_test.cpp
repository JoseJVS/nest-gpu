/*
 *  exact_geometry_test.cpp
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

// The mask/tile overlap filters against an exact oracle (geometry2d_oracle.h):
//   B2  Mask::tile_overlap of source/target masks, through the sub-tile tree
//   B1  Mask::tiles_within_mask_range of blueprints on tile pairs (also across
//       edge-wrap images) and Mask::overlap_with_tile_surface on node/tile
// They must accept every robust overlap and every tile (pair) holding a node
// (pair) the mask accepts, also within 1e-9 of the mask boundary. Rejecting
// gaps (tightness) is required for circular masks, whose filters are exact,
// and counted for the other shapes until their exact filters are enabled.

#include <cstdio>
#include <deque>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "bounding_box.h"
#include "creator_registry.h"
#include "geometry2d_oracle.h"
#include "geometry_oracle.h"
#include "geometry_test_helpers.h"
#include "gf2d_geometry.h"
#include "gf_collection.h"
#include "gf_creators.h"
#include "grid_generation.h"
#include "mask.h"
#include "thread_aligned_array.h"
#include "tile.h"
#include "type_erasure_helpers.h"
#include "vp_interface.h"

using namespace sapi;
using namespace geometry_test;

namespace {
using Oracle = GeometryOracle<Coord2D>;

// Whether the B1 and B2 filters of non-circular masks are exact (Q7.16d/e)
constexpr bool EXACT_NONCIRCULAR_B1 = false;
constexpr bool EXACT_NONCIRCULAR_B2 = false;

// Nodes this close to the mask boundary are sampled on both sides of it
constexpr double BOUNDARY_GAP = 1e-9;

const std::vector<MaskDef> MASKS{
    {MASK_SHAPE::CIRCULAR, {0.35}},
    {MASK_SHAPE::CIRCULAR, {1.5}},
    {MASK_SHAPE::ELLIPTICAL, {2., 0.2}},
    {MASK_SHAPE::ELLIPTICAL, {2., 0.2, 45.}},
    {MASK_SHAPE::ELLIPTICAL, {0.5, 1.2, -60.}},
    {MASK_SHAPE::ELLIPTICAL, {1.5, 0.015, 30.}}, // 1:100
    {MASK_SHAPE::PARALLELOGRAM, {1., 0.2, -0.3, 0.8}},
    {MASK_SHAPE::PARALLELOGRAM, {0.8, 0., 0.2, 0.4}},
    {MASK_SHAPE::PARALLELOGRAM, {1., 0., 1., 1e-2}}, // thin
    {MASK_SHAPE::TRIANGULAR, {1.2, 0., 0., 1.2}},
    {MASK_SHAPE::TRIANGULAR, {0., 0.9, 0.9, 0.}},     // clockwise
    {MASK_SHAPE::TRIANGULAR, {1.5, 0.1, -0.4, 0.3}},  // obtuse
    {MASK_SHAPE::TRIANGULAR, {2., 0.01, 1.9, 0.05}}}; // sliver

std::string shape_name(const MASK_SHAPE shape) {
  switch (shape) {
  case MASK_SHAPE::CIRCULAR:
    return "circular";
  case MASK_SHAPE::ELLIPTICAL:
    return "elliptical";
  case MASK_SHAPE::PARALLELOGRAM:
    return "parallelogram";
  default:
    return "triangular";
  }
}

std::string describe(const MaskDef &def, const P offset) {
  std::string s = shape_name(def.shape) + " {";
  for (const auto p : def.params)
    s += std::to_string(p) + " ";
  return s + "} at (" + std::to_string(offset.x) + ", " +
         std::to_string(offset.y) + ")";
}

// One tally per filter and mask shape; tightness is enforced where the filter
// is exact
std::map<std::string, FilterTally> tallies;

FilterTally &tally(const std::string &filter, const MASK_SHAPE shape,
                   const bool exact_noncircular) {
  const auto key = filter + " " + shape_name(shape);
  auto [it, inserted] = tallies.try_emplace(key);
  if (inserted) {
    it->second.name = key;
    it->second.enforce_tightness =
        shape == MASK_SHAPE::CIRCULAR || exact_noncircular;
  }
  return it->second;
}

FilterTally &b1_tally(const std::string &filter, const MASK_SHAPE shape) {
  return tally("B1 " + filter, shape, EXACT_NONCIRCULAR_B1);
}

FilterTally &b2_tally(const MASK_SHAPE shape) {
  return tally("B2 tile_overlap", shape, EXACT_NONCIRCULAR_B2);
}

Mask<Coord2D> source_mask(const MaskDef &def, const P origin,
                          const P offset = {0, 0}) {
  return construct_mask<Coord2D>(def.shape, {origin.x, origin.y}, def.params,
                                 {offset.x, offset.y});
}

Mask<Coord2D> blueprint(const MaskDef &def, const P offset) {
  return construct_mask<Coord2D>(def.shape, {}, def.params,
                                 {offset.x, offset.y});
}

// Reach of the shape from its anchor, from the oracle (not the code under test)
double extent(const MaskDef &def) { return Oracle::extent(Oracle::shape(def)); }

// A point strictly inside the shape, relative to the anchor
P interior_point(const Shape2D &shape) {
  if (shape.is_round())
    return {0, 0};
  P sum{0, 0};
  for (const auto &v : shape.vertices)
    sum = sum + v;
  return sum * (1. / static_cast<double>(shape.vertices.size()));
}

// Points within BOUNDARY_GAP of the boundary of the shape placed at anchor, on
// both sides, that lie in the polygon. They move along rays from an interior
// point, which cross every edge (also those through the anchor vertex).
std::vector<Coord2D> near_boundary(const Shape2D &shape, const P anchor,
                                   const std::vector<P> &polygon) {
  const P inner = interior_point(shape);
  std::vector<Coord2D> points;
  for (const auto &p : boundary_points(shape, 200)) {
    const P ray = (p - inner) * (BOUNDARY_GAP / norm(p - inner));
    for (const double side : {-1., 1.}) {
      const P q = anchor + p + ray * side;
      if (in_convex(q, polygon))
        points.emplace_back(to_coord(q));
    }
  }
  return points;
}

std::vector<Coord2D> with(std::vector<Coord2D> a,
                          const std::vector<Coord2D> &b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

std::vector<const Tile<Coord2D> *> roots_of(const std::vector<NamedTile> &c) {
  std::vector<const Tile<Coord2D> *> roots;
  for (const auto &nt : c)
    if (nt.name.find('.') == std::string::npos)
      roots.push_back(nt.tile);
  return roots;
}

// The oracle follows the parameter conventions of the node tests: points
// robustly inside (outside) the shape are accepted (rejected) by
// coord_in_mask, for source masks and blueprints. The two oracle deciders
// agree: the sampled one never finds an overlap the analytic one excludes.
void test_oracle() {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> unit(-1, 1);

  for (const auto &def : MASKS) {
    const auto shape = Oracle::shape(def);
    const P origin{0.3, -0.7}, offset{-0.2, 0.4};
    const auto mask = source_mask(def, origin, offset);
    const auto bp = blueprint(def, offset);
    const double reach = 1.2 * extent(def);
    const auto boundary = boundary_points(shape, 997);
    long decided = 0;
    for (int k = 0; k < 20000; ++k) {
      // Uniform in the bounding square, and close to the boundary
      const P d =
          k % 2 ? P{unit(rng), unit(rng)} * reach
                : boundary[static_cast<std::size_t>(k / 2) % boundary.size()] +
                      P{unit(rng), unit(rng)} * (1e-4 * reach);
      const int verdict = classify(shape, d);
      if (verdict == 0)
        continue;
      const bool in = verdict > 0;
      ++decided;
      const P at = origin + offset + d;
      require(mask.coord_in_mask(to_coord(at)).first == in,
              "oracle disagrees with coord_in_mask: " + describe(def, d));
      require(bp.coord_in_mask(to_coord(origin), to_coord(at)).first == in,
              "oracle disagrees with the blueprint: " + describe(def, d));
    }
    require(decided > 15000, "too few decided oracle samples");

    // Random triangles and quadrilaterals around the shape
    long robust_unsampled = 0, robust = 0;
    for (int k = 0; k < 2000; ++k) {
      std::vector<P> corners;
      const P centre = P{unit(rng), unit(rng)} * (1.5 * reach);
      for (int c = 0; c < 3 + k % 2; ++c)
        corners.emplace_back(centre + P{unit(rng), unit(rng)} * (0.5 * reach));
      const auto region = convex_hull(corners);
      if (region.size() < 3)
        continue;
      const auto v = oracle_verdict<Coord2D>(shape, region);
      const bool sampled = sampled_meets(shape, region, 400);
      require(v.possible || !sampled,
              "sampled overlap the analytic oracle excludes: " +
                  describe(def, centre));
      robust += v.robust;
      robust_unsampled += v.robust && !sampled;
    }
    // Sampling only misses slivers of overlap
    require(robust_unsampled * 50 <= robust,
            "the deciders disagree too often: " + describe(def, {0, 0}));
  }
}

// B2 through the sub-tile trees of the catalog, with node soundness
void test_source_target(const std::vector<NamedTile> &catalog) {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> unit(-1, 1);

  for (const auto *root : roots_of(catalog)) {
    const P centre = from(root->c_radius_.origin_);
    const double tile_radius = std::sqrt(root->c_radius_.radius2_);
    for (const auto &def : MASKS) {
      const auto shape = Oracle::shape(def);
      auto &t = b2_tally(def.shape);
      const double reach = tile_radius + extent(def);
      for (int k = 0; k < 24; ++k) {
        const P origin = centre + P{unit(rng), unit(rng)} * reach;
        const P offset = k % 3 ? P{0, 0} : P{unit(rng), unit(rng)} * 0.5;
        const auto mask = source_mask(def, origin, offset);
        const auto where = describe(def, origin + offset);
        check_tile_hierarchy(mask, shape, *root, t, where);

        // Node soundness on every tile of the tree
        std::vector<const Tile<Coord2D> *> stack{root};
        while (!stack.empty()) {
          const auto *tile = stack.back();
          stack.pop_back();
          check_tile_nodes(
              mask, *tile,
              with(Oracle::node_samples(*tile),
                   near_boundary(shape, origin + offset, polygon(*tile))),
              where);
          for (const auto &st : tile->sub_tiles_)
            stack.push_back(&st);
        }
      }
    }
  }
}

// B1 on catalog tile pairs (sub-tile pairs through the hierarchy), on node
// and tile, and the same configurations with the sides swapped: (A, B) under
// the blueprint S at offset o holds the displacements B - A - o, and (B, A)
// under -S at -o holds their negatives
void test_blueprints(const std::vector<NamedTile> &catalog) {
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> unit(-1, 1);
  const auto roots = roots_of(catalog);
  // Informational counters, printed but not asserted
  long asymmetric = 0, point_vs_mask = 0;

  for (const auto *a : roots)
    for (const auto *b : roots)
      for (const auto &def : MASKS) {
        const auto shape = Oracle::shape(def);
        const auto inverted_def = Oracle::reflected(def);
        const auto inverted_shape = Oracle::shape(inverted_def);
        const P ab = from(b->c_radius_.origin_) - from(a->c_radius_.origin_);
        const double reach = std::sqrt(a->c_radius_.radius2_) +
                             std::sqrt(b->c_radius_.radius2_) + extent(def);

        std::vector<P> offsets{ab, ab + P{reach, 0} * 0.98};
        for (int k = 0; k < 4; ++k)
          offsets.emplace_back(ab + P{unit(rng), unit(rng)} * reach);

        for (const auto &offset : offsets) {
          const auto where = describe(def, offset);
          const auto bp = blueprint(def, offset);
          check_pair_hierarchy(bp, shape, *a, *b, true,
                               b1_tally("tile pairs", def.shape), where);

          const auto inverted = blueprint(inverted_def, P{0, 0} - offset);
          check_pair_hierarchy(inverted, inverted_shape, *b, *a, true,
                               b1_tally("tile pairs (inverted)", def.shape),
                               where + " inverted");
          asymmetric += bp.tiles_within_mask_range(*a, *b) !=
                        inverted.tiles_within_mask_range(*b, *a);

          // Node pairs, also with b within BOUNDARY_GAP of the blueprint
          // boundary seen from each vertex of A
          const auto a_samples = Oracle::node_samples(*a);
          check_pair_nodes(bp, *a, *b, a_samples, Oracle::node_samples(*b),
                           where);
          for (const auto &v : a->vertices_)
            check_pair_nodes(
                bp, *a, *b, {v},
                near_boundary(shape, from(v) + offset, polygon(*b)), where);

          // A node of A drives the blueprint against B; the same placement
          // as a source mask on B
          for (std::size_t i = 0; i < a_samples.size(); i += 7) {
            const auto &c = a_samples[i];
            const bool accepted = check_point(
                bp, shape, c, *b, b1_tally("node/tile", def.shape), where);
            const auto mask = source_mask(def, from(c), offset);
            const bool b2 =
                check_tile_region(mask, shape, *b, b2_tally(def.shape),
                                  where + " (point placement)")
                    .level != OVERLAP_LEVEL::NONE;
            point_vs_mask += accepted != b2;
          }
        }
      }

  std::printf("B1 tile pairs: %ld verdicts change with the sides swapped\n",
              asymmetric);
  std::printf("B1 node/tile and B2 on the same placement disagree %ld times\n",
              point_vs_mask);
}

// B1 against the edge-wrap images of the pool tiles built by the grid
// functors: every image is checked as an ordinary pool tile, and its oracle
// verdict equals that of the unshifted pool tile under the shifted offset
void test_wrap_images() {
  TAArray<GFCollection<Coord2D>> gc_array;
  CreatorRegistry<GridTargetPositionShifts<Coord2D>> gscr;
  CreatorRegistry<ShiftedOriginCreator<Coord2D>> socr;
  CreatorRegistry<CachedTileCreator<Coord2D>> ctcr;
  initialize_gsc_registry(gscr);
  initialize_soc_registry(socr);
  initialize_ctc_registry(ctcr);

  struct GridDef {
    std::string shape;
    std::vector<tileidx_t> dims;
    std::vector<space_t> sides;
  };
  const std::vector<GridDef> grids{{"rectangle", {4, 3}, {1.}},
                                   {"rectangle", {3, 4}, {2., 1.}},
                                   {"triangle", {4, 4}, {1.}},
                                   {"hexagon", {4, 4}, {1.}}};
  const std::vector<MaskDef> defs{MASKS[1], MASKS[3], MASKS[5], MASKS[6],
                                  MASKS[11]};
  std::mt19937_64 rng(2026);
  std::uniform_real_distribution<double> unit(-1, 1);
  long images = 0;

  for (const auto &g : grids) {
    gc_array.clone(construct_gf_collection<Coord2D>(
        {0., 0.}, g.dims, g.shape, g.sides, {0}, gscr, socr, ctcr));
    auto grid = generate_tile_grid(gc_array);
    for (auto &pos : grid.positions_) {
      pos.tile_.split(1, false);
      for (auto &image : pos.tile_images_)
        if (image.shape_ != TILE_SHAPE::NULL_TS)
          image.split(1, false);
    }

    // Drivers at the first and the last grid position
    for (const auto *driver :
         {&grid.positions_.front(), &grid.positions_.back()})
      for (const auto &pool : grid.positions_)
        for (std::size_t i = 0; i < pool.tile_images_.size(); ++i) {
          const auto &image = pool.tile_images_[i].shape_ != TILE_SHAPE::NULL_TS
                                  ? pool.tile_images_[i]
                                  : pool.tile_;
          const P shift = from(pool.image_displacements_[i]);
          ++images;
          for (const auto &def : defs) {
            const auto shape = Oracle::shape(def);
            // Offsets reaching from the driver towards the image
            const P towards = from(image.c_radius_.origin_) -
                              from(driver->tile_.c_radius_.origin_);
            for (const P offset :
                 {towards, towards + P{unit(rng), unit(rng)} * extent(def)}) {
              const auto where = describe(def, offset) + " image " +
                                 std::to_string(i) + " of " + g.shape;
              const auto bp = blueprint(def, offset);
              check_pair_hierarchy(bp, shape, driver->tile_, image, true,
                                   b1_tally("tile pairs (images)", def.shape),
                                   where);

              const auto unshifted = blueprint(def, offset - shift);
              const auto v = oracle_verdict<Coord2D>(
                  shape, Oracle::pair_region(driver->tile_, image, bp.anchor_));
              const auto u = oracle_verdict<Coord2D>(
                  shape, Oracle::pair_region(driver->tile_, pool.tile_,
                                             unshifted.anchor_));
              require((!v.robust || u.possible) && (!u.robust || v.possible),
                      "image oracle differs from the shifted offset: " + where);
            }
          }
        }
  }
  require(images > 0, "no edge-wrap images");
}

// Touching and containing configurations with known answers. Touching sets
// share a boundary point, which the node tests accept, so every filter must
// accept them; the exact circular filters also reject gaps.
void test_analytic() {
  const auto square = [](const double x, const double y) {
    return Tile<Coord2D>(TILE_SHAPE::RECTANGLE, construct_coord_2D(x, y), {1.},
                         {});
  };
  const auto a = square(0.5, 0.5); // [0, 1]^2
  const auto b = square(1.5, 0.5); // shares an edge with a
  const auto c = square(3.5, 0.5); // gap of 2 to a
  const auto d = square(1.5, 1.5); // shares the vertex (1, 1) with a
  const MaskDef circle{MASK_SHAPE::CIRCULAR, {0.5}};
  const MaskDef ellipse{MASK_SHAPE::ELLIPTICAL, {2., 0.5}};
  const MaskDef ellipse90{MASK_SHAPE::ELLIPTICAL, {2., 0.5, 90.}};
  const MaskDef diamond{MASK_SHAPE::PARALLELOGRAM, {0.5, 0.5, -0.5, 0.5}};
  const MaskDef corner{MASK_SHAPE::TRIANGULAR, {1., 0., 0., 1.}};

  const auto overlaps = [](const Mask<Coord2D> &m, const Tile<Coord2D> &t) {
    return m.tile_overlap(t) != OVERLAP_LEVEL::NONE;
  };

  // Source masks tangent to the square [0, 1]^2
  require(overlaps(source_mask(circle, {1.5, 0.5}), a), "circle on edge");
  require(!overlaps(
              source_mask({MASK_SHAPE::CIRCULAR, {0.5 - 1e-6}}, {1.5, 0.5}), a),
          "circle short of edge");
  const P to_vertex = P{0.3, 0.4}; // length 0.5
  require(overlaps(source_mask(circle, P{1, 1} + to_vertex), a),
          "circle on vertex");
  require(!overlaps(source_mask(circle, P{1, 1} + to_vertex * (1 + 1e-6)), a),
          "circle short of vertex");
  require(overlaps(source_mask(ellipse, {0.5, 1.5}), a), "ellipse on edge");
  require(overlaps(source_mask(ellipse90, {0.5, 3.}), a),
          "rotated ellipse on edge");
  require(overlaps(source_mask(diamond, {1.5, 1.5}), a),
          "parallelogram corner on corner");
  require(overlaps(source_mask(corner, {1., 1.}), a),
          "triangle vertex on corner");
  require(overlaps(source_mask(corner, {0.5, 1.}), a),
          "triangle vertex on edge");

  // Containment
  const Tile<Coord2D> hexagon(TILE_SHAPE::HEXAGON, construct_coord_2D(0.2, 0.1),
                              {1.}, {30});
  require(source_mask({MASK_SHAPE::CIRCULAR, {0.1}}, {0.2, 0.1})
                  .tile_overlap(hexagon) == OVERLAP_LEVEL::PARTIAL,
          "mask in tile");
  require(source_mask({MASK_SHAPE::ELLIPTICAL, {3., 2., 20.}}, {0.5, 0.5})
                  .tile_overlap(a) == OVERLAP_LEVEL::FULL,
          "tile in mask");

  // Blueprints: A = B, shared edge and shared vertex hold the zero
  // displacement, which every shape holds
  for (const auto &def : MASKS) {
    require(blueprint(def, {0, 0}).tiles_within_mask_range(a, a), "A = B");
    require(blueprint(def, {0, 0}).tiles_within_mask_range(a, b),
            "shared edge");
    require(blueprint(def, {0, 0}).tiles_within_mask_range(a, d),
            "shared vertex");
    require(blueprint(def, {0, 0})
                .overlap_with_tile_surface(construct_coord_2D(1., 1.), d),
            "node on the tile vertex");
  }

  // c - a = [2, 4] x [-1, 1]: the vertex differences of the two squares are
  // collinear on its edges, e.g. (3, 1) mid-edge; a circle of radius 0.5
  // centred 0.5 above that point touches it
  require(blueprint(circle, {3., 1.5}).tiles_within_mask_range(a, c),
          "circle on collinear hull point");
  require(!blueprint({MASK_SHAPE::CIRCULAR, {0.5 - 1e-6}}, {3., 1.5})
               .tiles_within_mask_range(a, c),
          "circle short of collinear hull point");
  require(blueprint(circle, {1.5, 0.}).tiles_within_mask_range(a, c),
          "circle on hull edge");
  require(blueprint(ellipse, {0., 0.}).tiles_within_mask_range(a, c),
          "ellipse reaching hull edge");
  require(blueprint(corner, {1., -1.}).tiles_within_mask_range(a, c),
          "triangle vertex on hull vertex");
  require(blueprint(diamond, {1.5, 1.5}).tiles_within_mask_range(a, c),
          "parallelogram corner on hull vertex");

  // A gap of the analytic oracle stays a gap of the exact circular filter
  const auto v = oracle_verdict<Coord2D>(
      Oracle::shape({MASK_SHAPE::CIRCULAR, {0.5 - 1e-6}}),
      Oracle::pair_region(a, c, construct_coord_2D(3., 1.5)));
  require(!v.possible, "analytic gap");
}
} // namespace

int main() {
  init_omp();
  std::deque<Tile<Coord2D>> roots;
  const auto catalog = tile_catalog(roots, 2);

  test_oracle();
  test_analytic();
  test_source_target(catalog);
  test_blueprints(catalog);
  test_wrap_images();

  // Tightness counters (the over-accepted share is what the exact filters
  // of Q7.16d/e remove). Every tally sees robust overlaps and rejections, and
  // the conservative ones over-accept, so no check above holds vacuously.
  // B2 and the four B1 kinds, each for the four mask shapes
  require(tallies.size() == 20, "a filter tally is missing");
  for (const auto &[key, t] : tallies) {
    t.report();
    require(0 < t.robust && t.accepted < t.cases, key + " is one-sided");
    require(t.enforce_tightness || 0 < t.over_accepted,
            key + " never over-accepts");
  }
  return 0;
}
