/*
 *  geometry_oracle.h
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

#ifndef GEOMETRY_ORACLE_H
#define GEOMETRY_ORACLE_H

// Dimension-agnostic drivers that check the mask/tile overlap filters against
// an exact geometric oracle. The oracle itself is a per-dimension trait,
// GeometryOracle<CoordT>, specialized in geometry<D>d_oracle.h. It provides:
//   Shape                          exact description of a mask shape, built
//                                  from the mask parameters without SAPI code
//   Region                         convex set of displacements from the anchor
//   shape(def), reflected(def)     the shape of a definition, and of -shape
//   tile_region(tile, anchor)      tile - anchor (source/target masks)
//   pair_region(a, b, anchor)      b - a - anchor (blueprint tile pairs)
//   point_region(coord, b, anchor) b - coord - anchor (blueprint point/tile)
//   verdict(shape, region)         OracleVerdict of the shape meeting the
//                                  region (boundaries included)
//   extent(shape)                  largest distance from the anchor to the
//                                  shape
//   node_samples(tile)             interior, perimeter and vertex points

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

#include "mask.h"
#include "tile.h"

namespace geometry_test {
// Prints what failed and aborts; independent of NDEBUG
[[noreturn]] inline void fail(const std::string &what) {
  std::fprintf(stderr, "geometry oracle check failed: %s\n", what.c_str());
  std::abort();
}

inline void require(const bool condition, const std::string &what) {
  if (!condition)
    fail(what);
}

template <sapi::Coordinate CoordT> struct GeometryOracle;

struct MaskDef {
  sapi::MASK_SHAPE shape;
  std::vector<sapi::space_t> params;
};

// Exact verdicts come with a tolerance band of ORACLE_MARGIN relative to the
// shape size: a robust overlap persists when the shape shrinks by the margin,
// a possible overlap appears when it grows by it. Outside the band the exact
// verdict is unambiguous, so filters are judged only there.
inline constexpr double ORACLE_MARGIN = 1e-6;

struct OracleVerdict {
  bool robust = false;   // overlap of the shrunk shape
  bool possible = false; // overlap of the grown shape
};

template <sapi::Coordinate CoordT, typename Region>
OracleVerdict oracle_verdict(const typename GeometryOracle<CoordT>::Shape &s,
                             const Region &region) {
  const auto v = GeometryOracle<CoordT>::verdict(s, region);
  require(v.possible || !v.robust, "oracle verdicts are not nested");
  return v;
}

// Verdicts of one filter against the oracle. Soundness (a robust overlap is
// accepted) always holds; tightness (the filter rejects every pair without a
// possible overlap) holds only for exact filters, so it is counted and only
// required when enforce_tightness is set.
struct FilterTally {
  std::string name;
  bool enforce_tightness = false;
  long cases = 0;
  long accepted = 0;
  long robust = 0;
  long band = 0;          // inside the oracle's tolerance band
  long over_accepted = 0; // accepted without a possible overlap

  void record(const bool accepted_by_filter, const OracleVerdict v,
              const std::string &where) {
    ++cases;
    accepted += accepted_by_filter;
    robust += v.robust;
    band += v.possible && !v.robust;
    if (!accepted_by_filter && v.robust)
      fail(name + " rejects an overlap: " + where);
    if (accepted_by_filter && !v.possible) {
      ++over_accepted;
      if (enforce_tightness)
        fail(name + " accepts a gap: " + where);
    }
  }

  void report() const {
    std::printf("%-38s cases %8ld  accepted %8ld  robust %8ld  band %6ld  "
                "over-accepted %8ld (%.1f%% of accepted)\n",
                name.c_str(), cases, accepted, robust, band, over_accepted,
                accepted ? 100. * static_cast<double>(over_accepted) /
                               static_cast<double>(accepted)
                         : 0.);
  }
};

// Node-level soundness: any sample point the mask accepts makes the tile
// filter accept the tile
template <sapi::Coordinate CoordT>
void check_tile_nodes(const sapi::Mask<CoordT> &mask,
                      const sapi::Tile<CoordT> &tile,
                      const std::vector<CoordT> &samples,
                      const std::string &where) {
  if (mask.tile_overlap(tile) != sapi::OVERLAP_LEVEL::NONE)
    return;
  for (const auto &s : samples)
    if (mask.coord_in_mask(s).first)
      fail("tile_overlap rejects a tile holding a node in the mask: " + where);
}

// Node-level soundness of a blueprint: any accepted node pair makes the tile
// pair filter accept the pair
template <sapi::Coordinate CoordT>
void check_pair_nodes(const sapi::Mask<CoordT> &blueprint,
                      const sapi::Tile<CoordT> &a, const sapi::Tile<CoordT> &b,
                      const std::vector<CoordT> &a_samples,
                      const std::vector<CoordT> &b_samples,
                      const std::string &where) {
  if (blueprint.tiles_within_mask_range(a, b))
    return;
  for (const auto &sa : a_samples)
    for (const auto &sb : b_samples)
      if (blueprint.coord_in_mask(sa, sb).first)
        fail("tiles_within_mask_range rejects a pair holding a node pair in "
             "the blueprint: " +
             where);
}

struct TileCheck {
  sapi::OVERLAP_LEVEL level;
  OracleVerdict verdict;
};

// Records the tile_overlap verdict of a mask for one tile against the oracle
template <sapi::Coordinate CoordT>
TileCheck check_tile_region(const sapi::Mask<CoordT> &mask,
                            const typename GeometryOracle<CoordT>::Shape &shape,
                            const sapi::Tile<CoordT> &tile, FilterTally &tally,
                            const std::string &where) {
  const auto level = mask.tile_overlap(tile);
  const auto v = oracle_verdict<CoordT>(
      shape, GeometryOracle<CoordT>::tile_region(tile, mask.anchor_));
  tally.record(level != sapi::OVERLAP_LEVEL::NONE, v, where);
  return {level, v};
}

// Source/target mask hierarchy: every tile of the tree is checked against the
// oracle; a FULL parent implies every child possibly overlaps (monotonicity
// downwards), a robustly overlapping child implies its parent was accepted
// and, for leaves, that get_overlapping_leaf_sub_tiles reports it.
template <sapi::Coordinate CoordT>
void check_tile_hierarchy(const sapi::Mask<CoordT> &mask,
                          const typename GeometryOracle<CoordT>::Shape &shape,
                          const sapi::Tile<CoordT> &root, FilterTally &tally,
                          const std::string &where) {
  const auto leaves = mask.get_overlapping_leaf_sub_tiles(root);
  const std::unordered_set<const sapi::Tile<CoordT> *> reported(leaves.begin(),
                                                                leaves.end());

  struct Item {
    const sapi::Tile<CoordT> *tile;
    sapi::OVERLAP_LEVEL parent;
  };
  std::vector<Item> stack{{&root, sapi::OVERLAP_LEVEL::PARTIAL}};
  while (!stack.empty()) {
    const auto [tile, parent] = stack.back();
    stack.pop_back();

    const auto [level, v] = check_tile_region(mask, shape, *tile, tally, where);

    require(parent != sapi::OVERLAP_LEVEL::NONE || !v.robust,
            "a parent tile rejected a robustly overlapping child: " + where);
    require(parent != sapi::OVERLAP_LEVEL::FULL || v.possible,
            "a FULL parent holds a child outside the mask: " + where);
    if (tile->sub_tiles_.empty())
      require(!v.robust || reported.contains(tile),
              "get_overlapping_leaf_sub_tiles drops an overlapping leaf: " +
                  where);

    for (const auto &st : tile->sub_tiles_)
      stack.emplace_back(&st,
                         parent == sapi::OVERLAP_LEVEL::FULL ? parent : level);
  }
}

// Blueprint hierarchy, as walked by insert_leaf_tiles_within_range: a pair of
// sub-tiles is visited when its parent pair was accepted. A robustly
// overlapping pair is accepted at every level, so its parent pair was too.
template <sapi::Coordinate CoordT>
void check_pair_hierarchy(const sapi::Mask<CoordT> &blueprint,
                          const typename GeometryOracle<CoordT>::Shape &shape,
                          const sapi::Tile<CoordT> &a,
                          const sapi::Tile<CoordT> &b,
                          const bool parent_accepted, FilterTally &tally,
                          const std::string &where) {
  using Oracle = GeometryOracle<CoordT>;
  const bool accepted = blueprint.tiles_within_mask_range(a, b);
  const auto v = oracle_verdict<CoordT>(
      shape, Oracle::pair_region(a, b, blueprint.anchor_));
  tally.record(accepted, v, where);
  require(parent_accepted || !v.robust,
          "a parent pair rejected a robustly overlapping sub-tile pair: " +
              where);

  // Same-level sub-tile pairs, as for the leaves of the driver and pool tiles
  for (const auto &sa : a.sub_tiles_)
    for (const auto &sb : b.sub_tiles_)
      check_pair_hierarchy(blueprint, shape, sa, sb, accepted, tally, where);
}

// Blueprint driven by a single node: coord against the pool tile b
template <sapi::Coordinate CoordT>
bool check_point(const sapi::Mask<CoordT> &blueprint,
                 const typename GeometryOracle<CoordT>::Shape &shape,
                 const CoordT &coord, const sapi::Tile<CoordT> &b,
                 FilterTally &tally, const std::string &where) {
  using Oracle = GeometryOracle<CoordT>;
  const bool accepted = blueprint.overlap_with_tile_surface(coord, b);
  tally.record(accepted,
               oracle_verdict<CoordT>(
                   shape, Oracle::point_region(coord, b, blueprint.anchor_)),
               where);
  return accepted;
}
} // namespace geometry_test

#endif
