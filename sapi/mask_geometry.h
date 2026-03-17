/*
 *  mask_geometry.h
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

#ifndef MASK_GEOMETRY_H
#define MASK_GEOMETRY_H

#include <vector>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "enum_store.h"
#include "sapi_config.h"


namespace sapi
{
// Forward definition to coordinates.h
struct Coord2D;
struct Coord3D;
template < typename CoordT >
struct Displacement;

// Forward definition to mask.h
template < typename CoordT >
struct Mask;


void initialize_circular_mask_helpers(
    space_t& radius2,
    const std::vector< space_t >& mask_params
);


void initialize_elliptical_mask_helpers(
    space_t& radius2,
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< space_t >& mask_params
);


void initialize_parallelogram_mask_helpers(
    space_t& radius2,
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const Coord2D& origin,
    const std::vector< space_t >& mask_params
);


void initialize_triangular_mask_helpers(
    space_t& radius2,
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< space_t >& mask_params
);


template < typename CoordT >
inline void initialize_mask_helpers(
    space_t& radius2,
    std::vector< CoordT >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const CoordT& origin,
    const std::vector< space_t >& mask_params,
    const MASK_SHAPE& shape
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        const auto params_length = mask_params.size();
        switch ( shape )
        {
        case MASK_SHAPE::CIRCULAR:
        {
            if ( params_length != 1 )
                throw std::invalid_argument( "Invalid circular mask params vector" );

            initialize_circular_mask_helpers( radius2, mask_params );

            break;
        }

        case MASK_SHAPE::ELLIPTICAL:
        {
            if ( params_length < 2 || 3 < params_length )
                throw std::invalid_argument( "Invalid elliptical mask params vector" );

            initialize_elliptical_mask_helpers(
                radius2, helper_vectors, helper_scalars, mask_params
            );

            break;
        }

        case MASK_SHAPE::PARALLELOGRAM:
        {
            if ( params_length != 4 )
                throw std::invalid_argument( "Invalid parallelogram mask params vector" );

            initialize_parallelogram_mask_helpers(
                radius2, helper_vectors, helper_scalars, origin, mask_params
            );

            break;
        }

        case MASK_SHAPE::TRIANGULAR:
        {
            if ( params_length != 4 )
                throw std::invalid_argument( "Invalid triangular mask params vector" );

            initialize_triangular_mask_helpers(
                radius2, helper_vectors, helper_scalars, mask_params
            );

            break;
        }

        default:
            throw std::invalid_argument( "Invalid mask shape" );
        }
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}


std::optional< Displacement< Coord2D > >
    coord_in_circular_mask(
        const Mask< Coord2D >&,
        const Coord2D&
    );


std::optional< Displacement< Coord2D > >
    coord_in_elliptical_mask(
        const Mask< Coord2D >&,
        const Coord2D&
    );


std::optional< Displacement< Coord2D > >
    coord_in_parallelogram_mask(
        const Mask< Coord2D >&,
        const Coord2D&
    );


std::optional< Displacement< Coord2D > >
    coord_in_triangular_mask(
        const Mask< Coord2D >&,
        const Coord2D&
    );


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
    coord_in_mask(
        const Mask< CoordT >& mask,
        const CoordT& coord
    )
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( mask.shape_ )
        {
        case MASK_SHAPE::CIRCULAR:
            return coord_in_circular_mask( mask, coord );

        case MASK_SHAPE::ELLIPTICAL:
            return coord_in_elliptical_mask( mask, coord );

        case MASK_SHAPE::PARALLELOGRAM:
            return coord_in_parallelogram_mask( mask, coord );

        case MASK_SHAPE::TRIANGULAR:
            return coord_in_triangular_mask( mask, coord );

        default:
            throw std::invalid_argument( "Invalid mask shape" );
        }
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}


std::optional< Displacement< Coord2D > >
    coord_in_circular_mask(
        const Mask< Coord2D >&,
        const Coord2D&,
        const Coord2D&
    );


std::optional< Displacement< Coord2D > >
    coord_in_elliptical_mask(
        const Mask< Coord2D >&,
        const Coord2D&,
        const Coord2D&
    );


std::optional< Displacement< Coord2D > >
    coord_in_parallelogram_mask(
        const Mask< Coord2D >&,
        const Coord2D&,
        const Coord2D&
    );


std::optional< Displacement< Coord2D > >
    coord_in_triangular_mask(
        const Mask< Coord2D >&,
        const Coord2D&,
        const Coord2D&
    );


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
    coord_in_mask(
        const Mask< CoordT >& mask,
        const CoordT& a,
        const CoordT& b
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( mask.shape_ )
        {
        case MASK_SHAPE::CIRCULAR:
            return coord_in_circular_mask( mask, a, b );

        case MASK_SHAPE::ELLIPTICAL:
            return coord_in_elliptical_mask( mask, a, b );

        case MASK_SHAPE::PARALLELOGRAM:
            return coord_in_parallelogram_mask( mask, a, b );

        case MASK_SHAPE::TRIANGULAR:
            return coord_in_triangular_mask( mask, a, b );

        default:
            throw std::invalid_argument( "Invalid mask shape" );
        }
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}
}


#endif
