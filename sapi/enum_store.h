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
enum TILE_SHAPE
{
    NULL_TS,
    RECTANGLE,
    TRIANGLE,
    HEXAGON
};

enum MASK_SHAPE
{
    NULL_MS,
    CIRCULAR,
    ELLIPTICAL,
    PARALLELOGRAM,
    TRIANGULAR
};

enum OVERLAP_LEVEL
{
    NONE,
    PARTIAL,
    FULL
};

enum DISTRIBUTION_MODE
{
    FREE,
    SQUEEZED,
    BALANCED
};

enum CONNECTION_METHOD
{
    NULL_CM,
    PAIRWISE_BERNOULLI,
    PAIRWISE_POISSON,
    FIXED_NUMBER
};

enum UNARY_FUNCTION
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

enum DISPLACEMENT_FUNCTION
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
