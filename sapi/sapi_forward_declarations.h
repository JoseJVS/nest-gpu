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
#include <string>
#include <type_traits>
#include <vector>

#include "sapi_config.h"

namespace sapi {
// Interface shared by Coord2D and Coord3D, required from every CoordT.
// The *_vec members read or write D consecutive values and advance the
// caller's iterator by D, which a concept cannot express.
template <typename C>
concept Coordinate =
    std::default_initializable<C> && std::is_trivially_copyable_v<C> &&
    requires(const C c, const space_t s,
             std::vector<space_t>::const_iterator read_space_it,
             std::vector<space_t>::iterator write_space_it,
             std::vector<uint64_t>::const_iterator read_uint_it,
             std::vector<uint64_t>::iterator write_uint_it) {
      { C::D } -> std::convertible_to<dim_t>;
      { c + c } -> std::same_as<C>;
      { c + s } -> std::same_as<C>;
      { c - c } -> std::same_as<C>;
      { c - s } -> std::same_as<C>;
      { c * c } -> std::same_as<C>;
      { c * s } -> std::same_as<C>;
      { c / c } -> std::same_as<C>;
      { c / s } -> std::same_as<C>;
      { c == c } -> std::convertible_to<bool>;
      { c.is_null() } -> std::convertible_to<bool>;
      { c.negative() } -> std::same_as<C>;
      { c.sum() } -> std::same_as<space_t>;
      { c.product() } -> std::same_as<space_t>;
      { c.to_string() } -> std::same_as<std::string>;
      { c.copy_to_vec(write_space_it) } -> std::same_as<void>;
      { c.bit_copy_to_vec(write_uint_it) } -> std::same_as<void>;
      { C::copy_from_vec(read_space_it) } -> std::same_as<C>;
      { C::bit_copy_from_vec(read_uint_it) } -> std::same_as<C>;
    };

// Random engine result types supported by AnyRNG_T
template <typename RT>
concept RNGResult = std::same_as<RT, uint32_t> || std::same_as<RT, uint64_t>;

// Forward definition to tile.h
template <Coordinate CoordT> struct Tile;

// Forward definition to bounding_box.h
template <Coordinate CoordT> struct BoundingBox;

// Forward definitions to mask.h and mask_collection.h
template <Coordinate CoordT> struct Mask;
template <Coordinate CoordT> struct MaskCollection;

// Forward definitions to coordinates.h
template <Coordinate CoordT> struct Displacement;
template <Coordinate CoordT> struct CircumscribedRadius;

// Forward definitions to grid_functors.h
template <Coordinate CoordT> struct GridTargetPositionShifts;
template <Coordinate CoordT> struct ShiftedOriginCreator;
template <Coordinate CoordT> struct CachedTileCreator;

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
