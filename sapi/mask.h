/*
 *  mask.h
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

#ifndef MASK_H
#define MASK_H

#include "tile.h"
#include "mask_geometry.h"


namespace sapi
{
template < typename CoordT >
struct Mask
{
    MASK_SHAPE shape_ = MASK_SHAPE::NULL_MS;
    space_t radius2_ = 0;
    std::optional< CoordT > origin_;
    std::optional< CoordT > offset_;
    std::vector< CoordT > helper_vectors_;
    std::vector< space_t > helper_scalars_;

    Mask() = default;
    Mask( const Mask& ) = default;
    Mask( Mask&& ) = default;
    ~Mask() = default;

    Mask(
        const MASK_SHAPE&,
        const std::vector< space_t >&,
        const std::vector< space_t >&,
        const std::vector< space_t >&
    );

    Mask& operator=( Mask&& );

    std::optional< Displacement< CoordT > >
        coord_in_mask(
            const CoordT&
        ) const;

    std::optional< Displacement< CoordT > >
        coord_in_mask(
            const CoordT&,
            const CoordT&
        ) const;

    bool overlap_with_tile_surface(
        const Tile< CoordT >&
    ) const;

    bool overlap_with_tile_surface(
        const CoordT&,
        const Tile< CoordT >&
    ) const;

    bool tiles_within_mask_range(
        const Tile< CoordT >&,
        const Tile< CoordT >&
    ) const;

    OVERLAP_LEVEL tile_overlap(
        const Tile< CoordT >&
    ) const;

    std::forward_list< const Tile< CoordT >* >
        get_overlapping_leaf_sub_tiles(
            const Tile< CoordT >&
        ) const;
};


template < typename CoordT >
Mask< CoordT >::Mask(
    const MASK_SHAPE& shape,
    const std::vector< space_t >& mask_origin,
    const std::vector< space_t >& mask_params,
    const std::vector< space_t >& mask_offset
)
    : shape_( shape )
{
    if ( mask_params.empty() )
        throw std::invalid_argument( "Invalid mask params vector" );

    if ( !mask_origin.empty() && mask_origin.size() != static_cast< std::size_t >( CoordT::D ) )
        throw std::invalid_argument( "Invalid mask origin vector" );

    if ( !mask_offset.empty() && mask_offset.size() != static_cast< std::size_t >( CoordT::D ) )
        throw std::invalid_argument( "Invalid mask offset vector" );

    if ( !mask_offset.empty() )
        offset_.emplace( CoordT::copy_from_vec( mask_offset.begin() ) );

    if ( !mask_origin.empty() )
    {
        if ( offset_.has_value() )
            origin_.emplace( CoordT::copy_from_vec( mask_origin.begin() ) + offset_.value() );
        else
            origin_.emplace( CoordT::copy_from_vec( mask_origin.begin() ) );
    }

    initialize_mask_helpers(
        radius2_,
        helper_vectors_,
        helper_scalars_,
        origin_.value_or( CoordT() ),
        mask_params,
        shape_
    );
}


template < typename CoordT >
inline Mask< CoordT >&
Mask< CoordT >::operator=( Mask< CoordT >&& m )
{
    shape_ = m.shape_;
    radius2_ = m.radius2_;
    origin_ = std::move( m.origin_ );
    offset_ = std::move( m.offset_ );
    helper_vectors_ = std::move( m.helper_vectors_ );
    helper_scalars_ = std::move( m.helper_scalars_ );

    m.shape_ = MASK_SHAPE::NULL_MS;

    return *this;
}


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
Mask< CoordT >::coord_in_mask(
    const CoordT& coord
) const
{
    return sapi::coord_in_mask( *this, coord );
}


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
Mask< CoordT >::coord_in_mask(
    const CoordT& a,
    const CoordT& b
) const
{
    return sapi::coord_in_mask( *this, a, b );
}


template < typename CoordT >
inline bool Mask< CoordT >::overlap_with_tile_surface(
    const Tile< CoordT >& tile
) const
{
    return coord_in_mask( tile.project_point_to_surface( origin_.value() ) ).has_value();
}


template < typename CoordT >
inline bool Mask< CoordT >::overlap_with_tile_surface(
    const CoordT& coord,
    const Tile< CoordT >& tile
) const
{
    return coord_in_mask( coord, tile.project_point_to_surface( coord ) ).has_value();
}


template < typename CoordT >
inline bool Mask< CoordT >::tiles_within_mask_range(
    const Tile< CoordT >& a,
    const Tile< CoordT >& b
) const
{
    return coord_in_mask(
        a.project_point_to_surface(
            b.c_radius_.origin_
        ),
        b.project_point_to_surface(
            a.c_radius_.origin_
        )
    ).has_value();
}


template < typename CoordT >
OVERLAP_LEVEL Mask< CoordT >::tile_overlap(
    const Tile< CoordT >& tile
) const
{
    // Fast rejection check if mask origin is farther than sum of both radi
    if ( !tile.c_radius_.disp_in_radius( tile.c_radius_.origin_ - origin_.value(), std::sqrt( radius2_ ) ) )
        return OVERLAP_LEVEL::NONE;

    // Check if tile is fully in mask
    bool in_mask = true;
    vertidx_t vertices_in_mask = 0;
    auto tile_vit = tile.vertices_.cbegin();
    const auto tile_vend = tile.vertices_.cend();
    for ( ; tile_vit != tile_vend; ++tile_vit )
    {
        in_mask &= coord_in_mask( *tile_vit ).has_value();
        if ( !in_mask ) break; // Stop at first failure
        ++vertices_in_mask;
    }

    // If tile fully in mask
    if ( in_mask ) return OVERLAP_LEVEL::FULL;

    // If at least one vertex is in mask
    if ( 0 < vertices_in_mask ) return OVERLAP_LEVEL::PARTIAL;

    // If mask origin is in tile or vice versa
    if (
        tile.coord_in_tile( origin_.value() ) ||
        coord_in_mask( tile.c_radius_.origin_ ).has_value()
        )
        return OVERLAP_LEVEL::PARTIAL;

    // Continue with vertex check if possible
    // skipped if previous loop went over all vertices
    for ( ; tile_vit != tile_vend; ++tile_vit )
        if ( coord_in_mask( *tile_vit ).has_value() )
            return OVERLAP_LEVEL::PARTIAL;

    // Check for mask area overlap with tile edges
    if ( overlap_with_tile_surface( tile ) )
        return OVERLAP_LEVEL::PARTIAL;

    // Tile not in mask, no vertex in mask, mask not in tile, no edge overlap
    return OVERLAP_LEVEL::NONE;
}


template < typename CoordT >
std::forward_list< const Tile< CoordT >* >
Mask< CoordT >::get_overlapping_leaf_sub_tiles(
    const Tile< CoordT >& tile
) const
{
    std::forward_list< const Tile< CoordT >* > leaf_sub_tiles;
    std::forward_list< const Tile< CoordT >* > temps{ &tile };
    do
    {
        auto current = temps.front();
        temps.pop_front();

        // Get level of overlap with tile
        switch ( tile_overlap( *current ) )
        {
            // If the tile is completely within the mask we insert all of
            // its leaf sub tiles ( defaults to the tile in question if no sub tiles available )
        case OVERLAP_LEVEL::FULL:
            current->insert_leaf_sub_tiles( leaf_sub_tiles );
            break;

            // If there is only a partial overlap then we check into the sub-tiles
            // if there are no sub-tiles then we add the current tile to our result list
        case OVERLAP_LEVEL::PARTIAL:
            if ( !current->sub_tiles_.empty() )
            {
                for ( const auto& st : current->sub_tiles_ )
                    temps.emplace_front( &st );
            }
            else
                leaf_sub_tiles.emplace_front( std::move( current ) );
            break;

        default:
            break;
        }

    } while ( !temps.empty() );

    return leaf_sub_tiles;
}
}


#endif
