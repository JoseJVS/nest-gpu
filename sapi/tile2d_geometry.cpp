/*
 *  tile2d_geometry.cpp
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

#include <random>

#include "tile.h"
#include "tile2d_geometry.h"
#include "coordinate_geometry.h"


namespace sapi
{
void initialize_vertices_2D(
    std::vector< Coord2D >& vertices,
    CircumscribedRadius< Coord2D >& c_radius,
    const std::vector< space_t >& side_lengths,
    const angle_t& angular_offset,
    const bool& triangular_vertices
)
{
    assert( vertices.empty() );
    for ( const auto& side : side_lengths )
        if ( almost_zero( squared( side ) ) || std::signbit( side ) )
            throw std::invalid_argument( "Invalid 2D side length" );

    const auto half_width = side_lengths[ 0 ] / 2;
    const auto half_height = 1 < side_lengths.size() ? side_lengths[ 1 ] / 2 : half_width;

    const vertidx_t total_vertices = triangular_vertices ? 3 : 4;
    vertices.reserve( total_vertices );

    bool sign_width = false;
    bool sign_height = false;
    if ( angular_offset == 0 )
    {
        for ( vertidx_t vert = 0; vert < total_vertices; ++vert )
        {
            sign_width = vert == 1 || vert == 2;
            sign_height = 1 < vert;

            vertices.emplace_back(
                Coord2D(
                    compensated_sum( c_radius.origin_.x_, sign_width ? -half_width : half_width ),
                    compensated_sum( c_radius.origin_.y_, sign_height ? -half_height : half_height )
                )
            );
        }
    }
    else
    {
        const auto rotation = create_angular_offset( angular_offset );
        for ( vertidx_t vert = 0; vert < total_vertices; ++vert )
        {
            sign_width = vert == 1 || vert == 2;
            sign_height = 1 < vert;

            vertices.emplace_back(
                c_radius.origin_ + rotate_displacement(
                    Coord2D(
                        sign_width ? -half_width : half_width,
                        sign_height ? -half_height : half_height
                    ),
                    rotation
                )
            );
        }
    }

    c_radius.radius2_ = distance2( c_radius.origin_, vertices[ 0 ] );
    if ( almost_zero( c_radius.radius2_ ) )
        throw std::invalid_argument( "Invalid 2D radius" );
}


void initialize_hexagon_vertices(
    std::vector< Coord2D >& vertices,
    CircumscribedRadius< Coord2D >& c_radius,
    const space_t& side_length,
    const angle_t& angular_offset
)
{
    assert( vertices.empty() );

    c_radius.radius2_ = squared( side_length );
    if ( std::signbit( side_length ) || almost_zero( c_radius.radius2_ ) )
        throw std::invalid_argument( "Invalid hexagon side length" );

    vertices.reserve( 6 );
    for ( angle_t angle_ix = 0; angle_ix < 6; ++angle_ix )
        vertices.emplace_back(
            c_radius.origin_ + ( create_angular_offset(
                angle_ix * 60 + angular_offset
            ) * side_length )
        );
}


void initialize_helpers_2D(
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< Coord2D >& vertices
)
{
    assert( ( vertices.size() == 3 || vertices.size() == 4 ) &&
        helper_vectors.empty() && helper_scalars.empty() );

    helper_vectors.reserve( 2 );
    helper_vectors.emplace_back( vertices[ 0 ] - vertices[ 1 ] );
    helper_vectors.emplace_back( vertices[ 2 ] - vertices[ 1 ] );

    helper_scalars.resize( 1, coord_sum(
        vector_cross( helper_vectors[ 0 ], helper_vectors[ 1 ] )
    ) );

    assert( !almost_zero( helper_scalars[ 0 ] ) );
}


void initialize_hexagon_helpers(
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< Coord2D >& vertices
)
{
    assert( vertices.size() == 6 &&
        helper_vectors.empty() &&
        helper_scalars.empty() );

    // Hexagon is divided into rectangles composed of opposing edges
    // used for parallelogram projection comparison
    helper_vectors.reserve( 6 );
    for ( vertidx_t vix = 0; vix < 3; ++vix )
    {
        helper_vectors.emplace_back(
            vertices[ vix + 1 ] - vertices[ vix ]
        );
        helper_vectors.emplace_back(
            vertices[ ( vix + 4 ) % 6 ] - vertices[ vix ]
        );
    }

    helper_scalars.resize( 1, coord_sum(
        vector_cross( helper_vectors[ 0 ], helper_vectors[ 1 ] )
    ) );

    assert( !almost_zero( helper_scalars[ 0 ] ) );
}


void split_rectangle(
    std::vector< Tile< Coord2D > >& sub_tiles,
    const std::vector< Coord2D >& vertices,
    const CircumscribedRadius< Coord2D >& c_radius,
    const tileidx_t& index,
    const split_t& splits
)
{
    if ( splits < 1 || !sub_tiles.empty() )
        return;

    sub_tiles.reserve( 4 );
    const split_t rem_splits = splits - 1;
    const space_t radius2 = c_radius.radius2_ / 4;

    Coord2D midpoints[ 4 ];
    for ( vertidx_t vix = 0; vix < 4; ++vix )
        midpoints[ vix ] = midpoint( vertices[ vix ], vertices[ ( vix + 1 ) % 4 ] );

    Coord2D new_origin;
    vertidx_t opposite_midpoint;
    for ( vertidx_t partition = 0; partition < 4; ++partition )
    {
        opposite_midpoint = ( partition + 3 ) % 4;
        new_origin = midpoint( c_radius.origin_, vertices[ partition ] );

        sub_tiles.emplace_back(
            Tile< Coord2D >(
                TILE_SHAPE::RECTANGLE,
                compute_sub_tile_split_index( index, partition, 5, rem_splits ),
                CircumscribedRadius< Coord2D >(
                    new_origin,
                    radius2
                ),
                { vertices[ partition ],
                midpoints[ partition ],
                c_radius.origin_,
                midpoints[ opposite_midpoint ] }
            )
        );
    }

#pragma omp taskloop num_tasks( 4 ) mergeable final( rem_splits < 4 )\
default( none ) shared( sub_tiles ) firstprivate( rem_splits )
    for ( vertidx_t partition = 0; partition < 4; ++partition )
    {
        const auto st_it = sub_tiles.begin() + partition;
        initialize_helpers_2D(
            st_it->helper_vectors_,
            st_it->helper_scalars_,
            st_it->vertices_
        );
        split_rectangle(
            st_it->sub_tiles_,
            st_it->vertices_,
            st_it->c_radius_,
            st_it->index_,
            rem_splits
        );
    }
}


void split_triangle(
    std::vector< Tile< Coord2D > >& sub_tiles,
    const std::vector< Coord2D >& vertices,
    const tileidx_t& index,
    const split_t& splits
)
{
    if ( splits < 1 || !sub_tiles.empty() )
        return;

    sub_tiles.reserve( 2 );
    const split_t rem_splits = splits - 1;

    // Get AB, the largest edge of triangle ABC
    Coord2D coordA, coordB, coordC;
    {
        std::array< Coord2D, 3 > sorted_coords;
        std::array< Coord2D, 3 > sorted_vectors;
        std::array< space_t, 3 > sorted_distances2;
        sort_vertices(
            { vertices[ 0 ], vertices[ 1 ], vertices[ 2 ] },
            sorted_coords,
            sorted_vectors,
            sorted_distances2
        );
        coordA = std::move( sorted_coords[ 0 ] );
        coordB = std::move( sorted_coords[ 1 ] );
        coordC = std::move( sorted_coords[ 2 ] );
    }

    // Get D midpoint of AB,
    // this midpoint is now a vertex of each sub-triangle DCA and DCB,
    // this guarantees each sub-triangle has same area size.
    const Coord2D coordD( midpoint( coordA, coordB ) );

    sub_tiles.emplace_back(
        Tile< Coord2D >(
            TILE_SHAPE::TRIANGLE,
            index + 1,
            compute_c_radius( coordC, coordD, coordA ),
            { coordC,
            coordD,
            coordA }
        )
    );
    sub_tiles.emplace_back(
        Tile< Coord2D >(
            TILE_SHAPE::TRIANGLE,
            compute_sub_tile_split_index( index, 1, 3, rem_splits ),
            compute_c_radius( coordC, coordD, coordB ),
            { coordC,
            coordD,
            coordB }
        )
    );

#pragma omp taskloop num_tasks( 2 ) mergeable final( rem_splits < 8 )\
default( none ) shared( sub_tiles ) firstprivate( rem_splits )
    for ( vertidx_t partition = 0; partition < 2; ++partition )
    {
        const auto st_it = sub_tiles.begin() + partition;
        initialize_helpers_2D(
            st_it->helper_vectors_,
            st_it->helper_scalars_,
            st_it->vertices_
        );
        split_triangle(
            st_it->sub_tiles_,
            st_it->vertices_,
            st_it->index_,
            rem_splits
        );
    }
}


void split_hexagon(
    std::vector< Tile< Coord2D > >& sub_tiles,
    const std::vector< Coord2D >& vertices,
    const CircumscribedRadius< Coord2D >& c_radius,
    const tileidx_t& index,
    const split_t& splits
)
{
    if ( splits < 1 || !sub_tiles.empty() )
        return;

    sub_tiles.reserve( 6 );
    const split_t rem_splits = splits - 1;
    const space_t radius2 = c_radius.radius2_ / 3;

    Coord2D new_origin;
    vertidx_t next_partition;
    for ( vertidx_t partition = 0; partition < 6; ++partition )
    {
        next_partition = ( partition + 1 ) % 6;

        new_origin = ( c_radius.origin_ / 3 )
            + ( ( vertices[ partition ] + vertices[ next_partition ] ) / 3 );

        sub_tiles.emplace_back(
            Tile< Coord2D >(
                TILE_SHAPE::TRIANGLE,
                compute_sub_tile_split_index( index, partition, 3, rem_splits ),
                CircumscribedRadius< Coord2D >(
                    new_origin,
                    radius2
                ),
                { c_radius.origin_,
                vertices[ partition ],
                vertices[ next_partition ] }
            )
        );
    }

#pragma omp taskloop num_tasks( 3 ) mergeable final( rem_splits < 9 )\
default( none ) shared( sub_tiles ) firstprivate( rem_splits )
    for ( vertidx_t partition = 0; partition < 6; ++partition )
    {
        const auto st_it = sub_tiles.begin() + partition;
        initialize_helpers_2D(
            st_it->helper_vectors_,
            st_it->helper_scalars_,
            st_it->vertices_
        );
        split_triangle(
            st_it->sub_tiles_,
            st_it->vertices_,
            st_it->index_,
            rem_splits
        );
    }
}


bool coord_in_tile_2D(
    const Coord2D& coord,
    const CircumscribedRadius< Coord2D >& c_radius,
    const std::vector< Coord2D >& vertices,
    const std::vector< Coord2D >& helper_vectors,
    const std::vector< space_t >& helper_scalars,
    const bool& triangular_projection
)
{
    if ( !c_radius.coord_in_radius( coord ).has_value() )
        return false;

    // Only works for squares or triangles
    return algebraic_projection_comparison(
        coord - vertices[ 1 ],
        helper_vectors[ 0 ],
        helper_vectors[ 1 ],
        helper_scalars[ 0 ],
        triangular_projection
    );
}


bool coord_in_hexagon(
    const Coord2D& coord,
    const CircumscribedRadius< Coord2D >& c_radius,
    const std::vector< Coord2D >& vertices,
    const std::vector< Coord2D >& helper_vectors,
    const std::vector< space_t >& helper_scalars
)
{
    if ( !c_radius.coord_in_radius( coord ).has_value() )
        return false;

    // Parallelogram based comparison using rotating rectangles
    bool res = false;
    vertidx_t rel_vix;
    for ( vertidx_t vix = 0; vix < 3; ++vix )
    {
        rel_vix = 2 * vix;
        res = algebraic_projection_comparison(
            coord - vertices[ vix ],
            helper_vectors[ rel_vix ],
            helper_vectors[ rel_vix + 1 ],
            helper_scalars[ 0 ],
            false // Triangular comparison
        );
        if ( res ) break;
    }

    return res;
}


void generate_coords_in_rectangle(
    std::vector< Coord2D >& coord_vec,
    AnyRNG& rng,
    const std::vector< Coord2D >& vertices,
    const std::vector< Coord2D >& helper_vectors
)
{
    space_t m0, m1;
    std::uniform_real_distribution< space_t > dist( 0., 1. );
    for ( auto coord_vec_it = coord_vec.begin();
        coord_vec_it != coord_vec.end(); ++coord_vec_it )
    {
        m0 = static_cast< space_t >( dist( rng ) );
        m1 = static_cast< space_t >( dist( rng ) );
        *coord_vec_it = Coord2D(
            compensated_sum(
                vertices[ 1 ].x_,
                helper_vectors[ 0 ].x_ * m0,
                helper_vectors[ 1 ].x_ * m1
            ),
            compensated_sum(
                vertices[ 1 ].y_,
                helper_vectors[ 0 ].y_ * m0,
                helper_vectors[ 1 ].y_ * m1
            )
        );
    }
}


void generate_coords_in_triangle(
    std::vector< Coord2D >& coord_vec,
    AnyRNG& rng,
    const std::vector< Coord2D >& vertices
)
{
    space_t m0, m1, m2;
    std::uniform_real_distribution< space_t > dist( 0., 1. );
    for ( auto coord_vec_it = coord_vec.begin();
        coord_vec_it != coord_vec.end(); ++coord_vec_it )
    {
        m0 = static_cast< space_t >( dist( rng ) );
        m1 = static_cast< space_t >( dist( rng ) );
        if ( std::isless( 1., m0 + m1 ) )
        {
            m0 = std::fmax( std::fmin( 1., 1. - m0 ), 0. );
            m1 = std::fmax( std::fmin( 1., 1. - m1 ), 0. );
        }
        m2 = std::fmax( std::fmin( 1., 1. - m0 - m1 ), 0. );
        *coord_vec_it = Coord2D(
            compensated_sum(
                vertices[ 0 ].x_ * m0,
                vertices[ 1 ].x_ * m1,
                vertices[ 2 ].x_ * m2
            ),
            compensated_sum(
                vertices[ 0 ].y_ * m0,
                vertices[ 1 ].y_ * m1,
                vertices[ 2 ].y_ * m2
            )
        );
    }
}


void generate_coords_in_hexagon(
    std::vector< Coord2D >& coord_vec,
    AnyRNG& rng,
    const std::vector< Coord2D >& vertices,
    const CircumscribedRadius< Coord2D >& c_radius
)
{
    vertidx_t hex_idx;
    space_t m0, m1, m2;
    std::uniform_real_distribution< space_t > space_dist( 0., 1. );
    std::uniform_int_distribution< vertidx_t > partition_dist( 0, 5 );
    for ( auto coord_vec_it = coord_vec.begin();
        coord_vec_it != coord_vec.end(); ++coord_vec_it )
    {
        m0 = static_cast< space_t >( space_dist( rng ) );
        m1 = static_cast< space_t >( space_dist( rng ) );
        if ( std::isless( 1., m0 + m1 ) )
        {
            m0 = std::fmax( std::fmin( 1., 1. - m0 ), 0. );
            m1 = std::fmax( std::fmin( 1., 1. - m1 ), 0. );
        }
        m2 = std::fmax( std::fmin( 1., 1. - m0 - m1 ), 0. );
        hex_idx = partition_dist( rng );
        *coord_vec_it = Coord2D(
            compensated_sum(
                c_radius.origin_.x_ * m0,
                vertices[ hex_idx ].x_ * m1,
                vertices[ ( hex_idx + 1 ) % 6 ].x_ * m2
            ),
            compensated_sum(
                c_radius.origin_.y_ * m0,
                vertices[ hex_idx ].y_ * m1,
                vertices[ ( hex_idx + 1 ) % 6 ].y_ * m2
            )
        );
    }
}


Coord2D project_point_to_2D_perimeter(
    const Coord2D& coord,
    const std::vector< Coord2D >& vertices
)
{
    vertidx_t v_index = 0;
    vertidx_t first = 0;
    space_t d_first = std::numeric_limits< space_t >::max();
    vertidx_t second = 0;
    space_t d_second = std::numeric_limits< space_t >::max();
    for ( const auto& vertex : vertices )
    {
        const auto d2 = distance2( coord, vertex );
        if ( std::isless( d2, d_first ) )
        {
            second = first;
            d_second = d_first;
            first = v_index;
            d_first = d2;
        }
        else if ( std::isless( d2, d_second ) )
        {
            second = v_index;
            d_second = d2;
        }
        ++v_index;
    }

    if ( almost_zero( d_first ) )
        return vertices[ first ];

    return projection_coord(
        coord,
        vertices[ first ],
        vertices[ second ],
        true // Clamped projection
    );
}
}
