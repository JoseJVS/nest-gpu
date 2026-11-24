/*
 *  gf_creators.cpp
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

#include <stdexcept>

#include "creator_registry.h"
#include "gf_creators.h"

// Registries of the dimensions without an implementation yet; the 2D ones are
// in gf2d_creators.cpp
namespace sapi {
void initialize_gsc_registry(
    CreatorRegistry<GridTargetPositionShifts<Coord3D>> &) {
  throw std::runtime_error("3D GridGenerator not yet implemented");
}

void initialize_soc_registry(CreatorRegistry<ShiftedOriginCreator<Coord3D>> &) {
  throw std::runtime_error("3D GridGenerator not yet implemented");
}

void initialize_ctc_registry(CreatorRegistry<CachedTileCreator<Coord3D>> &) {
  throw std::runtime_error("3D GridGenerator not yet implemented");
}
} // namespace sapi
