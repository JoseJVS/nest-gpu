/*
 *  gf2d_geometry.h
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

#ifndef GF2D_GEOMETRY_H
#define GF2D_GEOMETRY_H

#include "grid_functors.h"


namespace sapi
{
//Forward definition to link with coordinate_geometry.h
Coord2D create_angular_offset( const space_t& );
Coord2D rotate_displacement( const Coord2D&, const Coord2D& );


inline void initialize_soc_2D(
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< space_t >& side_lengths,
    const angle_t& angular_offset
)
{
    helper_scalars.reserve( 2 );
    helper_scalars.emplace_back( side_lengths[ 0 ] );
    helper_scalars.emplace_back( 1 < side_lengths.size() ? side_lengths[ 1 ] : side_lengths[ 0 ] );

    if ( angular_offset != 0 )
        helper_vectors.resize( 1, create_angular_offset( angular_offset ) );
}


inline void initialize_hexagonal_soc(
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< space_t >& side_lengths,
    const angle_t& angular_offset
)
{
    helper_scalars.reserve( 3 );
    // With side length a, height h of eq triangle is equal to sqrt( 3 ) * a / 2
    // To reach neighboring hexagon center we need 2 * h
    helper_scalars.emplace_back( std::sqrt( space_t( 3 ) ) * side_lengths[ 0 ] );
    helper_scalars.emplace_back( helper_scalars[ 0 ] / 2 );
    helper_scalars.emplace_back( 3 * side_lengths[ 0 ] / 2 );

    if ( angular_offset != 0 )
        helper_vectors.resize( 1, create_angular_offset( angular_offset ) );
}


inline Coord2D create_shifted_rectangular_origin(
    const ShiftedOriginCreator< Coord2D >& soc,
    const Coord2D& grid_origin,
    const GridPosition< Coord2D >& grid_position
)
{
    Coord2D disp_vector(
        soc.helper_scalars_[ 0 ] * grid_position[ 0 ],
        soc.helper_scalars_[ 1 ] * grid_position[ 1 ]
    );
    if ( !soc.helper_vectors_.empty() )
        return grid_origin + rotate_displacement( disp_vector, soc.helper_vectors_[ 0 ] );
    else
        return grid_origin + disp_vector;
}


inline Coord2D create_shifted_triangular_origin(
    const ShiftedOriginCreator< Coord2D >& soc,
    const Coord2D& grid_origin,
    const GridPosition< Coord2D >& grid_position,
    const GridPositionParity< Coord2D >& grid_position_parity
)
{
    Coord2D disp_vector(
        soc.helper_scalars_[ 0 ] * ( grid_position_parity[ 0 ] ? grid_position[ 0 ] / 2 : ( grid_position[ 0 ] - 1 ) / 2 ),
        soc.helper_scalars_[ 1 ] * grid_position[ 1 ]
    );
    if ( !soc.helper_vectors_.empty() )
        return grid_origin + rotate_displacement( disp_vector, soc.helper_vectors_[ 0 ] );
    else
        return grid_origin + disp_vector;
}


inline Coord2D create_shifted_hexagonal_origin(
    const ShiftedOriginCreator< Coord2D >& soc,
    const Coord2D& grid_origin,
    const GridPosition< Coord2D >& grid_position,
    const GridPositionParity< Coord2D >& grid_position_parity
)
{
    Coord2D disp_vector(
        grid_position_parity[ 1 ]
        ? soc.helper_scalars_[ 0 ] * grid_position[ 0 ]
        : compensated_sum( soc.helper_scalars_[ 1 ], soc.helper_scalars_[ 0 ] * grid_position[ 0 ] ),
        soc.helper_scalars_[ 2 ] * grid_position[ 1 ]
    );
    if ( !soc.helper_vectors_.empty() )
        return grid_origin + rotate_displacement( disp_vector, soc.helper_vectors_[ 0 ] );
    else
        return grid_origin + disp_vector;
}


inline void initialize_rectangular_ctc(
    std::vector< space_t >& side_lengths_,
    std::vector< angle_t >& angular_offsets_,
    const std::vector< space_t >& side_lengths,
    const angle_t& angular_offset
)
{
    side_lengths_ = side_lengths;
    angular_offsets_.resize( 1, angular_offset );
}


inline void initialize_triangular_ctc(
    std::vector< space_t >& side_lengths_,
    std::vector< angle_t >& angular_offsets_,
    const std::vector< space_t >& side_lengths,
    const angle_t& angular_offset
)
{
    side_lengths_ = side_lengths;
    angular_offsets_.reserve( 2 );
    angular_offsets_.emplace_back( angular_offset );
    angular_offsets_.emplace_back( angular_offset + 180 );
}


inline void initialize_hexagonal_ctc(
    std::vector< space_t >& side_lengths_,
    std::vector< angle_t >& angular_offsets_,
    const std::vector< space_t >& side_lengths,
    const angle_t& angular_offset
)
{
    side_lengths_ = side_lengths;
    angular_offsets_.resize( 1, angular_offset + 30 );
}


inline bool check_rectangular_dimensions(
    const CachedTileCreator< Coord2D >& ctc,
    const GridPosition< Coord2D >& grid_dimensions
)
{
    if ( ( grid_dimensions[ 0 ] + grid_dimensions[ 1 ] > 2 )
        && ( ctc.angular_offsets_[ 0 ] % 45 != 0 ) )
        return false;

    return true;
}


inline bool check_triangular_dimensions(
    const CachedTileCreator< Coord2D >& ctc,
    const GridPosition< Coord2D >& grid_dimensions
)
{
    const bool many_tiles = grid_dimensions[ 0 ] + grid_dimensions[ 1 ] > 2;
    if ( !many_tiles )
        return true;
    if ( many_tiles && ( ctc.angular_offsets_[ 1 ] % 180 != 0 ) )
        return false;
    if ( many_tiles && ( grid_dimensions[ 0 ] % 2 != 0 ) )
        return false;

    return true;
}


inline bool check_hexagonal_dimensions(
    const CachedTileCreator< Coord2D >& ctc,
    const GridPosition< Coord2D >& grid_dimensions
)
{
    const bool many_tiles = grid_dimensions[ 0 ] + grid_dimensions[ 1 ] > 2;
    if ( !many_tiles )
        return true;
    if ( many_tiles && ( ctc.angular_offsets_[ 0 ] % 30 != 0 ) )
        return false;
    if ( many_tiles && ( grid_dimensions[ 1 ] % 2 != 0 ) )
        return false;

    return true;
}


inline Tile< Coord2D > create_rectangle(
    const CachedTileCreator< Coord2D >& ctc,
    const Coord2D& tile_origin
)
{
    return Tile< Coord2D >( TILE_SHAPE::RECTANGLE, tile_origin, ctc.side_lengths_, ctc.angular_offsets_ );
}


inline Tile< Coord2D > create_triangle(
    const CachedTileCreator< Coord2D >& ctc,
    const Coord2D& tile_origin,
    const GridPositionParity< Coord2D >& grid_position_status
)
{
    return Tile< Coord2D >( TILE_SHAPE::TRIANGLE, tile_origin, ctc.side_lengths_,
        { grid_position_status[ 0 ] ? ctc.angular_offsets_[ 0 ] : ctc.angular_offsets_[ 1 ] } );
}


inline Tile< Coord2D > create_hexagon(
    const CachedTileCreator< Coord2D >& ctc,
    const Coord2D& tile_origin
)
{
    return Tile< Coord2D >( TILE_SHAPE::HEXAGON, tile_origin, ctc.side_lengths_, ctc.angular_offsets_ );
}
}


#endif
