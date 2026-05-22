/*
 *  enum_store.h
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

#ifndef ENUM_STORE_H
#define ENUM_STORE_H

namespace sapi
{
enum class TILE_SHAPE : uint8_t
{
    NULL_TS,
    RECTANGLE,
    TRIANGLE,
    HEXAGON
};

enum class MASK_SHAPE : uint8_t
{
    NULL_MS,
    CIRCULAR,
    ELLIPTICAL,
    PARALLELOGRAM,
    TRIANGULAR
};

enum class OVERLAP_LEVEL : uint8_t
{
    NONE,
    PARTIAL,
    FULL
};

enum class DISTRIBUTION_MODE : uint8_t
{
    FREE,
    SQUEEZED,
    BALANCED
};

enum class CONNECTION_METHOD : uint8_t
{
    NULL_CM,
    PAIRWISE_BERNOULLI,
    PAIRWISE_POISSON,
    FIXED_IN_DEGREE,
    FIXED_OUT_DEGREE
};

enum class UNARY_FUNCTION : uint8_t
{
    NULL_UF,
    IDENTITY,
    MIN,
    MAX,
    LOWER_BOUND,
    UPPER_BOUND,
    INVERSE,
    FACTOR,
    OFFSET,
    EXPONENTIAL,
    GAUSSIAN
};

enum class DISPLACEMENT_FUNCTION : uint8_t
{
    NULL_DF,
    CONSTANT,
    DISTANCE,
    DISPLACEMENT_X,
    DISPLACEMENT_Y,
    DISPLACEMENT_Z,
    DISTANCE_X,
    DISTANCE_Y,
    DISTANCE_Z
};
}

#endif
