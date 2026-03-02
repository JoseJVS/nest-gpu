/*
 *  bounding_box.h
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

#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <vector>

#include "coordinate_geometry.h"


namespace sapi
{
constexpr bool coord_interval_test(
    const Coord2D& value, const std::pair< Coord2D, Coord2D >& minmax
)
{
    return interval_test( minmax.first.x_, value.x_, minmax.second.x_ ) &&
        interval_test( minmax.first.y_, value.y_, minmax.second.y_ );
}


constexpr bool coord_interval_test(
    const Coord3D& value, const std::pair< Coord3D, Coord3D >& minmax
)
{
    return interval_test( minmax.first.x_, value.x_, minmax.second.x_ ) &&
        interval_test( minmax.first.y_, value.y_, minmax.second.y_ ) &&
        interval_test( minmax.first.z_, value.z_, minmax.second.z_ );
}


template < typename CoordT >
struct BoundingBox
{
    // As all indexes are defined positives,
    // -1 acts as a sentinel for boxes unmapped to
    // grid positions
    tileidx_t tile_index_ = -1;

    // Boxes are defined as pair of minimum and
    // maximum coordinates
    std::pair< CoordT, CoordT > minmax_bounds_;

    // Bounding box is defined regardless of rotation
    // such that inner boxes bounds are with local bounds
    // minx < x < maxx
    // miny < y < maxy
    // (minz < z < maxz)
    std::vector< BoundingBox< CoordT > > inner_boxes_;

    BoundingBox() = default;
    BoundingBox( const BoundingBox& ) = delete;
    BoundingBox( BoundingBox&& ) = default;
    ~BoundingBox() = default;

    BoundingBox(
        std::pair< CoordT, CoordT >&& minmax_bounds,
        const tileidx_t& tile_index
    )
        : tile_index_( tile_index )
        , minmax_bounds_( std::move( minmax_bounds ) )
    {
    }

    BoundingBox(
        std::pair< CoordT, CoordT >&& minmax_bounds,
        std::vector< BoundingBox< CoordT > >&& inner_boxes
    )
        : minmax_bounds_( std::move( minmax_bounds ) )
        , inner_boxes_( std::move( inner_boxes ) )
    {
    }

    bool coord_in_box( const CoordT& coord ) const
    {
        return coord_interval_test( coord, minmax_bounds_ );
    }

    BoundingBox& operator=( BoundingBox&& bb )
    {
        tile_index_ = bb.tile_index_;
        bb.tile_index_ = -1;

        minmax_bounds_ = std::move( bb.minmax_bounds_ );
        inner_boxes_ = std::move( bb.inner_boxes_ );

        return *this;
    }
};


template < typename CoordT >
BoundingBox< CoordT >
stack_boxes(
    std::vector< BoundingBox< CoordT > >&& bbs
)
{
    const auto bb_count = bbs.size();
    assert( bb_count == 1 || bb_count % 2 == 0 );

    if ( bb_count == 1 )
        return std::move( bbs[ 0 ] );

    auto [min, max] = minmax_coords< CoordT >();

    for ( const auto& bb : bbs )
    {
        update_min( min, bb.minmax_bounds_.first );
        update_max( max, bb.minmax_bounds_.second );
    }

    return BoundingBox< CoordT >(
        std::make_pair( std::move( min ), std::move( max ) ),
        std::move( bbs )
    );
}
}


#endif
