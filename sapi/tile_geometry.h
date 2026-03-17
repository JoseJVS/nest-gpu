/*
 *  tile_geometry.h
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

#ifndef TILE_GEOMETRY_H
#define TILE_GEOMETRY_H

#include <cmath>

#include "enum_store.h"
#include "tile2d_geometry.h"


namespace sapi
{
template < typename CoordT >
inline void
    initialize_tile_vertices(
        std::vector< CoordT >& vertices,
        CircumscribedRadius< CoordT >& c_radius,
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

            initialize_vertices_2D(
                vertices,
                c_radius,
                side_lengths,
                angular_offsets.empty()
                ? 0
                : angular_offsets[ 0 ],
                false // triangular vertices
            );

            break;
        }

        case TILE_SHAPE::TRIANGLE:
        {
            if ( lengths < 1 || 2 < lengths )
                throw std::invalid_argument( "Invalid triangle length vector" );

            initialize_vertices_2D(
                vertices,
                c_radius,
                side_lengths,
                angular_offsets.empty()
                ? 0
                : angular_offsets[ 0 ],
                true // triangular vertices
            );

            break;
        }

        case TILE_SHAPE::HEXAGON:
        {
            if ( lengths != 1 )
                throw std::invalid_argument( "Invalid hexagon length vector" );

            initialize_hexagon_vertices(
                vertices,
                c_radius,
                side_lengths[ 0 ],
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


template < typename CoordT >
inline void initialize_tile_helpers(
    std::vector< CoordT >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< CoordT >& vertices,
    const TILE_SHAPE& shape
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( shape )
        {
        case TILE_SHAPE::RECTANGLE:
        {
            initialize_helpers_2D(
                helper_vectors,
                helper_scalars,
                vertices
            );

            break;
        }

        case TILE_SHAPE::TRIANGLE:
        {
            initialize_helpers_2D(
                helper_vectors,
                helper_scalars,
                vertices
            );

            break;
        }

        case TILE_SHAPE::HEXAGON:
        {
            initialize_hexagon_helpers(
                helper_vectors,
                helper_scalars,
                vertices
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


template < typename CoordT >
inline void split_tile(
    std::vector< Tile< CoordT > >& sub_tiles,
    const std::vector< CoordT >& vertices,
    const CircumscribedRadius< CoordT >& c_radius,
    const tileidx_t& index,
    const split_t& splits,
    const TILE_SHAPE& shape
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( shape )
        {
        case TILE_SHAPE::RECTANGLE:
        {
            split_rectangle(
                sub_tiles,
                vertices,
                c_radius,
                index,
                splits
            );

            break;
        }

        case TILE_SHAPE::TRIANGLE:
        {
            split_triangle(
                sub_tiles,
                vertices,
                index,
                splits
            );

            break;
        }

        case TILE_SHAPE::HEXAGON:
        {
            split_hexagon(
                sub_tiles,
                vertices,
                c_radius,
                index,
                splits
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


template < typename CoordT >
inline bool coord_in_tile(
    const CoordT& coord,
    const CircumscribedRadius< CoordT >& c_radius,
    const std::vector< CoordT >& vertices,
    const std::vector< CoordT >& helper_vectors,
    const std::vector< space_t >& helper_scalars,
    const TILE_SHAPE& shape
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( shape )
        {
        case TILE_SHAPE::RECTANGLE:
            return coord_in_tile_2D(
                coord,
                c_radius,
                vertices,
                helper_vectors,
                helper_scalars,
                false
            );

        case TILE_SHAPE::TRIANGLE:
            return coord_in_tile_2D(
                coord,
                c_radius,
                vertices,
                helper_vectors,
                helper_scalars,
                true
            );

        case TILE_SHAPE::HEXAGON:
            return coord_in_hexagon(
                coord,
                c_radius,
                vertices,
                helper_vectors,
                helper_scalars
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


template < typename CoordT >
inline void generate_coords_in_tile(
    std::vector< CoordT >& coord_vec,
    AnyRNG& rng,
    const std::vector< CoordT >& vertices,
    const std::vector< CoordT >& helper_vectors,
    const CircumscribedRadius< CoordT >& c_radius,
    const TILE_SHAPE& shape
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        switch ( shape )
        {
        case TILE_SHAPE::RECTANGLE:
        {
            generate_coords_in_rectangle(
                coord_vec,
                rng,
                vertices,
                helper_vectors
            );

            break;
        }

        case TILE_SHAPE::TRIANGLE:
        {
            generate_coords_in_triangle(
                coord_vec,
                rng,
                vertices
            );

            break;
        }

        case TILE_SHAPE::HEXAGON:
        {
            generate_coords_in_hexagon(
                coord_vec,
                rng,
                vertices,
                c_radius
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


template < typename CoordT >
inline CoordT project_point_to_surface(
    const CoordT& coord,
    const std::vector< CoordT >& vertices
)
{
    if constexpr ( std::is_same_v< CoordT, Coord2D > )
    {
        return project_point_to_2D_perimeter(
            coord,
            vertices
        );
    }
    else
    {
        throw std::runtime_error( "3D implementation not available yet" );
    }
}


inline tileidx_t
    compute_sub_tile_count( const split_t& splits, const TILE_SHAPE& shape )
{
    switch ( shape )
    {
    case TILE_SHAPE::RECTANGLE:
        return std::pow( 4, splits );

    case TILE_SHAPE::TRIANGLE:
        return std::pow( 2, splits );

    case TILE_SHAPE::HEXAGON:
        return splits > 0 ? 6 * std::pow( 2, splits - 1 ) : 1;

    default:
        throw std::invalid_argument( "Invalid tile shape" );
    }
}


inline std::vector< split_t >
    get_possible_sub_tile_branches( const split_t& splits, const TILE_SHAPE& shape )
{
    if ( splits == 0 )
        return {};

    switch ( shape )
    {
    case TILE_SHAPE::RECTANGLE:
        return std::vector< split_t >( splits, 4 );

    case TILE_SHAPE::TRIANGLE:
        return std::vector< split_t >( splits, 2 );

    case TILE_SHAPE::HEXAGON:
    {
        std::vector< split_t > split_vec( splits, 2 );
        split_vec[ 0 ] = 6;
        return split_vec;
    }

    default:
        throw std::invalid_argument( "Invalid tile shape" );
    }
}


inline std::string
    get_name( const TILE_SHAPE& shape )
{
    switch ( shape )
    {
    case TILE_SHAPE::RECTANGLE:
        return "Rectangle";

    case TILE_SHAPE::TRIANGLE:
        return "Triangle";

    case TILE_SHAPE::HEXAGON:
        return "Hexagon";

    default:
        throw std::invalid_argument( "Invalid tile shape" );
    }
}
}


#endif
