/*
 *  sapi_config.h
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

#ifndef SAPI_CONFIG_H
#define SAPI_CONFIG_H

#include <array>
#include <cstdint>

#include "config.h"

namespace sapi {
typedef int32_t vp_t;      // Virtual process id type
typedef int32_t nodeidx_t; // Needs to be signed to handle overflow
typedef int64_t
    largenodeidx_t; // Needs to be at least equal or larger than nodeidx_t
typedef int32_t tileidx_t; // Needs to be signed to handle overflow
typedef uint8_t
    vertidx_t; // Current implemented tiles count with up to 12 vertices
typedef uint8_t
    split_t; // Needs to be unsigned and exponentially smaller than tileidx_t
typedef uint8_t dim_t;   // For static coordinate dimension count
typedef int8_t shift_t;  // Needs to be signed and smaller than tileidx_t
typedef int32_t count_t; // Needs to be signed
typedef int16_t angle_t; // Integer angular values are needed for grid coherence
typedef double space_t;  // For spatial data
typedef uint32_t conn_index_t; // For connection indexes
typedef float conn_param_t;    // For connection parameters
typedef uint32_t
    rng_bits_t; // Determines random engine result size can be 32 or 64 bits
typedef uint32_t rng_seed_t; // RNG seed type fixed u32bit with type
typedef uint64_t
    combined_idx_t; // Type for storing bitwise combination of two 32bit indexes

// Numerics constants
constexpr static const uint8_t TOLERANCE = 2;
constexpr static const uint8_t RELATIVE_TOLERANCE = 8;
// Relative margin by which conservative mask filters (bounding discs, exact
// tile tests) are widened: far above the rounding slack of the node-level
// tests, far below any real gap, so a filter never rejects what the node test
// accepts
constexpr static const space_t EXACT_FILTER_REL_TOL = 0x1p-30;
// Iteration cap of the GJK intersection query (convex_intersection.h):
// polygons finish in a few iterations, curved shapes converge geometrically;
// a query that reaches the cap reports an intersection (conservative)
constexpr static const uint8_t GJK_MAX_ITERATIONS = 64;

// Grid constants
// Grid periods a tile is shifted by along each dimension to form its images
// under edge wrap; a tile has IMAGE_SHIFTS.size()^D images
constexpr static const std::array<shift_t, 3> IMAGE_SHIFTS = {-1, 0, 1};

// Task constants
// Tasks per thread when many small jobs are grouped into chunk tasks: enough
// for the runtime to balance chunks of unequal duration, few enough that the
// task overhead stays small
constexpr static const uint8_t CHUNKS_PER_THREAD = 4;
// Node distribution batching across tasks threshold
constexpr static const uint8_t BATCHING_THRESHOLD = 100;

// RNG constants
constexpr static const char *const DEFAULT_RNG_TYPE_ = "pcg";
constexpr static const rng_seed_t DEFAULT_BASE_SEED_ = 143202461;
constexpr static const rng_seed_t RANK_SEEDER_ = 0xc229212d;
constexpr static const rng_seed_t THREAD_SEEDER_ = 0x37722d5e;
constexpr static const rng_seed_t PARITY_SEEDER_ = 0xb84c9bae;
} // namespace sapi

#endif
