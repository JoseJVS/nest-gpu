/*
 *  gf_geometry.h
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

#ifndef GF_GEOMETRY_H
#define GF_GEOMETRY_H

#include <stdexcept>

#include "enum_store.h"
#include "coordinates.h"
#include "grid_containers.h"


namespace sapi
{
// Forward definitions to tile.h
template < typename CoordT >
struct Tile;

// Forward definitions to grid_functors.h
template < typename CoordT >
struct ShiftedOriginCreator;
template < typename CoordT >
struct CachedTileCreator;


void initialize_soc_2D(
    std::vector< Coord2D >&,
    std::vector< space_t >&,
    const std::vector< space_t >&,
    const angle_t&
);


void initialize_hexagonal_soc(
    std::vector< Coord2D >&,
    std::vector< space_t >&,
    const std::vector< space_t >&,
    const angle_t&
);


template < typename CoordT >
inline void initialize_soc(
    std::vector< CoordT >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< space_t >& side_lengths,
    const std::vector< angle_t >& angular_offsets,
    const TILE_SHAPE& shape
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        if ( 1 < angular_offsets.size() )
            throw std::invalid_argument( "Invalid offsets for tile 2D" );
        const auto lengths = side_lengths.size();

        switch ( shape )
        {
        case TILE_SHAPE::RECTANGLE:
        {
            if ( lengths < 1 || 2 < lengths )
                throw std::invalid_argument( "Invalid rectangle length vector" );

            initialize_soc_2D(
                helper_vectors,
                helper_scalars,
                side_lengths,
                angular_offsets.empty()
                ? 0
                : angular_offsets[ 0 ]
            );

            break;
        }

        case TILE_SHAPE::TRIANGLE:
        {
            if ( lengths < 1 || 2 < lengths )
                throw std::invalid_argument( "Invalid triangle length vector" );

            initialize_soc_2D(
                helper_vectors,
                helper_scalars,
                side_lengths,
                angular_offsets.empty()
                ? 0
                : angular_offsets[ 0 ]
            );

            break;
        }

        case TILE_SHAPE::HEXAGON:
        {
            if ( lengths != 1 )
                throw std::invalid_argument( "Invalid hexagon length vector" );

            initialize_hexagonal_soc(
                helper_vectors,
                helper_scalars,
                side_lengths,
                angular_offsets.empty()
                ? 0
                : angular_offsets[ 0 ]
            );

            break;
        }

        default:
            throw std::invalid_argument( "Invalid tile shape" );
        }
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}


Coord2D create_shifted_rectangular_origin(
    const ShiftedOriginCreator< Coord2D >&,
    const Coord2D&,
    const GridPosition< Coord2D >&
);


Coord2D create_shifted_triangular_origin(
    const ShiftedOriginCreator< Coord2D >&,
    const Coord2D&,
    const GridPosition< Coord2D >&,
    const GridPositionParity< Coord2D >&
);


Coord2D create_shifted_hexagonal_origin(
    const ShiftedOriginCreator< Coord2D >&,
    const Coord2D&,
    const GridPosition< Coord2D >&,
    const GridPositionParity< Coord2D >&
);


template < typename CoordT >
inline CoordT create_shifted_origin(
    const ShiftedOriginCreator< CoordT >& soc,
    const CoordT& grid_origin,
    const GridPosition< CoordT >& gp,
    const GridPositionParity< CoordT >& gpp
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( soc.shape_ )
        {
        case TILE_SHAPE::RECTANGLE:
            return create_shifted_rectangular_origin(
                soc,
                grid_origin,
                gp
            );

        case TILE_SHAPE::TRIANGLE:
            return create_shifted_triangular_origin(
                soc,
                grid_origin,
                gp,
                gpp
            );

        case TILE_SHAPE::HEXAGON:
            return create_shifted_hexagonal_origin(
                soc,
                grid_origin,
                gp,
                gpp
            );

        default:
            throw std::invalid_argument( "Invalid tile shape" );
        }
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}


void initialize_rectangular_ctc(
    std::vector< space_t >&,
    std::vector< angle_t >&,
    const std::vector< space_t >&,
    const angle_t&
);


void initialize_triangular_ctc(
    std::vector< space_t >&,
    std::vector< angle_t >&,
    const std::vector< space_t >&,
    const angle_t&
);


void initialize_hexagonal_ctc(
    std::vector< space_t >&,
    std::vector< angle_t >&,
    const std::vector< space_t >&,
    const angle_t&
);


inline void initialize_ctc(
    std::vector< space_t >& side_lengths_,
    std::vector< angle_t >& angular_offsets_,
    const std::vector< space_t >& side_lengths,
    const std::vector< angle_t >& angular_offsets,
    const TILE_SHAPE& shape
)
{
    if ( 1 < angular_offsets.size() )
        throw std::invalid_argument( "Invalid offsets for tile 2D" );
    const auto lengths = side_lengths.size();

    switch ( shape )
    {
    case TILE_SHAPE::RECTANGLE:
    {
        if ( lengths < 1 || 2 < lengths )
            throw std::invalid_argument( "Invalid rectangle length vector" );

        initialize_rectangular_ctc(
            side_lengths_,
            angular_offsets_,
            side_lengths,
            angular_offsets.empty()
            ? 0
            : angular_offsets[ 0 ]
        );

        break;
    }

    case TILE_SHAPE::TRIANGLE:
    {
        if ( lengths < 1 || 2 < lengths )
            throw std::invalid_argument( "Invalid triangle length vector" );

        initialize_triangular_ctc(
            side_lengths_,
            angular_offsets_,
            side_lengths,
            angular_offsets.empty()
            ? 0
            : angular_offsets[ 0 ]
        );

        break;
    }

    case TILE_SHAPE::HEXAGON:
    {
        if ( lengths != 1 )
            throw std::invalid_argument( "Invalid hexagon length vector" );

        initialize_hexagonal_ctc(
            side_lengths_,
            angular_offsets_,
            side_lengths,
            angular_offsets.empty()
            ? 0
            : angular_offsets[ 0 ]
        );

        break;
    }

    default:
        throw std::invalid_argument( "Invalid tile shape" );
    }
}


bool check_rectangular_dimensions(
    const CachedTileCreator< Coord2D >&,
    const GridPosition< Coord2D >&
);


bool check_triangular_dimensions(
    const CachedTileCreator< Coord2D >&,
    const GridPosition< Coord2D >&
);


bool check_hexagonal_dimensions(
    const CachedTileCreator< Coord2D >&,
    const GridPosition< Coord2D >&
);


template < typename CoordT >
inline bool check_dimensions(
    const CachedTileCreator< CoordT >& ctc,
    const GridPosition< CoordT >& grid_dimensions
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( ctc.shape_ )
        {
        case TILE_SHAPE::RECTANGLE:
            return check_rectangular_dimensions(
                ctc,
                grid_dimensions
            );

        case TILE_SHAPE::TRIANGLE:
            return check_triangular_dimensions(
                ctc,
                grid_dimensions
            );

        case TILE_SHAPE::HEXAGON:
            return check_hexagonal_dimensions(
                ctc,
                grid_dimensions
            );

        default:
            throw std::invalid_argument( "Invalid tile shape" );
        }
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}


Tile< Coord2D > create_rectangle(
    const CachedTileCreator< Coord2D >&,
    const Coord2D&
);


Tile< Coord2D > create_triangle(
    const CachedTileCreator< Coord2D >&,
    const Coord2D&,
    const GridPositionParity< Coord2D >&
);


Tile< Coord2D > create_hexagon(
    const CachedTileCreator< Coord2D >&,
    const Coord2D&
);


template < typename CoordT >
inline Tile< CoordT > create_tile(
    const CachedTileCreator< CoordT >& ctc,
    const CoordT& tile_origin,
    const GridPositionParity< CoordT >& gpp
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( ctc.shape_ )
        {
        case TILE_SHAPE::RECTANGLE:
            return create_rectangle(
                ctc,
                tile_origin
            );

        case TILE_SHAPE::TRIANGLE:
            return create_triangle(
                ctc,
                tile_origin,
                gpp
            );

        case TILE_SHAPE::HEXAGON:
            return create_hexagon(
                ctc,
                tile_origin
            );

        default:
            throw std::invalid_argument( "Invalid tile shape" );
        }
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}
}


#endif
