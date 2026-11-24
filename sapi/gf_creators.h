/*
 *  gf_creators.h
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

#ifndef GF_CREATORS_H
#define GF_CREATORS_H

#include <vector>

#include "coordinates.h"
#include "creator_registry.h"
#include "grid_functors.h"
#include "sapi_forward_declarations.h"
#include "type_erasure_helpers.h"

namespace sapi {
template <Coordinate CoordT, TILE_SHAPE shape>
struct SOCCreator final
    : public StateLessCreator<ShiftedOriginCreator<CoordT>> {
  ShiftedOriginCreator<CoordT>
  create(const std::vector<space_t> &side_lengths,
         const std::vector<angle_t> &angular_offsets) const override {
    return construct_shifted_origin_creator<CoordT>(shape, side_lengths,
                                                    angular_offsets);
  }
};

template <Coordinate CoordT, TILE_SHAPE shape>
struct CTCCreator final : public StateLessCreator<CachedTileCreator<CoordT>> {
  CachedTileCreator<CoordT>
  create(const std::vector<space_t> &side_lengths,
         const std::vector<angle_t> &angular_offsets) const override {
    return construct_cached_tile_creator<CoordT>(shape, side_lengths,
                                                 angular_offsets);
  }
};

// Registries per dimension: gf2d_creators.cpp fills the 2D ones
void initialize_gsc_registry(
    CreatorRegistry<GridTargetPositionShifts<Coord2D>> &gscr);
void initialize_gsc_registry(
    CreatorRegistry<GridTargetPositionShifts<Coord3D>> &gscr);
void initialize_soc_registry(
    CreatorRegistry<ShiftedOriginCreator<Coord2D>> &socr);
void initialize_soc_registry(
    CreatorRegistry<ShiftedOriginCreator<Coord3D>> &socr);
void initialize_ctc_registry(CreatorRegistry<CachedTileCreator<Coord2D>> &ctcr);
void initialize_ctc_registry(CreatorRegistry<CachedTileCreator<Coord3D>> &ctcr);
} // namespace sapi

#endif
