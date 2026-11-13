/*
 *  sapi_forward_declarations.h
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

#ifndef SAPI_FORWARD_DECLARATIONS_H
#define SAPI_FORWARD_DECLARATIONS_H

#include <concepts>
#include <cstdint>
#include <type_traits>

#include "sapi_config.h"

namespace sapi {
// Random engine result types supported by AnyRNG_T
template <typename RT>
concept RNGResult = std::same_as<RT, uint32_t> || std::same_as<RT, uint64_t>;

// Forward definition to tile.h
template <typename CoordT> struct Tile;

// Forward definition to bounding_box.h
template <typename CoordT> struct BoundingBox;

// Forward definitions to mask.h and mask_collection.h
template <typename CoordT> struct Mask;
template <typename CoordT> struct MaskCollection;

// Forward definitions to coordinates.h
template <typename CoordT> struct Displacement;
template <typename CoordT> struct CircumscribedRadius;

// Forward definitions to grid_functors.h
template <typename CoordT> struct GridTargetPositionShifts;
template <typename CoordT> struct ShiftedOriginCreator;
template <typename CoordT> struct CachedTileCreator;

// Forward definitions to type_erasure_helpers.h and creator_registry.h
template <typename RT> struct StateLessCreator;
template <typename RT> class CreatorRegistry;

// Forward definitions to vp_interface.h
template <typename T> struct OmpLock;
template <typename T> struct OmpLockGuard;

// Forward definition to thread_aligned_array.h
template <typename T>
  requires std::is_copy_constructible_v<T>
class TAArray;

// Forward definition to type_erasure_helpers.h
template <RNGResult RT> class AnyRNG_T;
using AnyRNG = AnyRNG_T<rng_bits_t>;
} // namespace sapi

#endif
