/*
 *  gf_collection.h
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

#ifndef GF_COLLECTION_H
#define GF_COLLECTION_H

#include <algorithm>

#include "grid_functors.h"
#include "creator_registry.h"


namespace sapi
{
template < typename CoordT >
class GFCollection : public Clonable< GFCollection< CoordT > >
{
public:
    GFCollection() = delete;
    GFCollection( const GFCollection& ) = delete;
    GFCollection( GFCollection&& ) = default;

    GFCollection(
        const CoordT&,
        const GridPosition< CoordT >&,
        const std::unique_ptr< GridTargetPositionShifts< CoordT > >&,
        const std::unique_ptr< BaseShiftedOriginCreator< CoordT > >&,
        const std::unique_ptr< BaseCachedTileCreator< CoordT > >&
    );

    GFCollection(
        const std::vector< space_t >& grid_origin,
        const std::vector< tileidx_t >& grid_dimensions,
        const std::string& tile_type,
        const std::vector< space_t >& tile_params,
        const CreatorRegistry< GridTargetPositionShifts< CoordT > >& gscr,
        const CreatorRegistry< BaseShiftedOriginCreator< CoordT > >& socr,
        const CreatorRegistry< BaseCachedTileCreator< CoordT > >& ctcr
    );

    GFCollection& operator=( GFCollection&& );

    bool is_initialized() const;
    bool check_dimensions( const bool& edge_wrap ) const;
    std::unique_ptr< GFCollection > clone() const override;

    const CoordT& get_grid_origin() const;
    const GridPosition< CoordT >& get_grid_dimensions() const;
    const std::unique_ptr< GridTargetPositionShifts< CoordT > >&
        get_grid_shifts() const;

    tileidx_t get_index( const GridPosition< CoordT >& ) const;

    CoordT shift_origin(
        const GridPosition< CoordT >&,
        const GridPositionParity< CoordT >&
    ) const;

    std::unique_ptr< Tile< CoordT > > create_tile(
        const CoordT&,
        const GridPositionParity< CoordT >&
    ) const;

protected:
    CoordT grid_origin_;
    GridPosition< CoordT > grid_dimensions_;
    std::unique_ptr< GridTargetPositionShifts< CoordT > > gps_;
    std::unique_ptr< BaseShiftedOriginCreator< CoordT > > soc_;
    std::unique_ptr< BaseCachedTileCreator< CoordT > > ctc_;
};


template < typename CoordT >
GFCollection< CoordT >::GFCollection(
    const CoordT& grid_origin,
    const GridPosition< CoordT >& grid_dimensions,
    const std::unique_ptr< GridTargetPositionShifts< CoordT > >& gps,
    const std::unique_ptr< BaseShiftedOriginCreator< CoordT > >& soc,
    const std::unique_ptr< BaseCachedTileCreator< CoordT > >& ctc
)
{
    if ( !gps || !soc || !ctc )
        throw std::invalid_argument( "Cannot construct GFCollection using nullptr" );

    grid_origin_ = grid_origin;
    grid_dimensions_ = grid_dimensions;
    gps_ = gps->clone();
    soc_ = soc->clone();
    ctc_ = ctc->clone();
}


template < typename CoordT >
GFCollection< CoordT >::GFCollection(
    const std::vector< space_t >& grid_origin,
    const std::vector< tileidx_t >& grid_dimensions,
    const std::string& tile_type,
    const std::vector< space_t >& tile_params,
    const CreatorRegistry< GridTargetPositionShifts< CoordT > >& gscr,
    const CreatorRegistry< BaseShiftedOriginCreator< CoordT > >& socr,
    const CreatorRegistry< BaseCachedTileCreator< CoordT > >& ctcr
)
{
    if (
        grid_origin.empty() ||
        grid_origin.size() != static_cast< std::size_t >( CoordT::D ) ||
        grid_dimensions.size() != grid_origin.size() ||
        !std::all_of( grid_dimensions.cbegin(), grid_dimensions.cend(), positiveTix )
        )
        throw std::invalid_argument( "Invalid grid params for GFCollection" );

    std::copy(
        grid_dimensions.cbegin(),
        grid_dimensions.cend(),
        grid_dimensions_.begin()
    );

    grid_origin_ = CoordT::copy_from_vec( grid_origin.begin() );
    gps_ = gscr.get_creator( tile_type )->create();
    soc_ = socr.get_creator( tile_type )->create( tile_params );
    ctc_ = ctcr.get_creator( tile_type )->create( tile_params );
}


template < typename CoordT >
inline GFCollection< CoordT >&
GFCollection< CoordT >::operator=( GFCollection&& gfc )
{
    grid_origin_ = std::move( gfc.grid_origin_ );
    grid_dimensions_ = std::move( gfc.grid_dimensions_ );
    gps_ = std::move( gfc.gps_ );
    soc_ = std::move( gfc.soc_ );
    ctc_ = std::move( gfc.ctc_ );
}


template < typename CoordT >
inline bool GFCollection< CoordT >::is_initialized() const
{
    return gps_ && soc_ && ctc_;
}


template < typename CoordT >
inline bool GFCollection< CoordT >::check_dimensions( const bool& edge_wrap ) const
{
    return ctc_->check_dimensions(
        grid_dimensions_,
        edge_wrap
    );
}


template < typename CoordT >
inline std::unique_ptr< GFCollection< CoordT > >
GFCollection< CoordT >::clone() const
{
    return std::make_unique< GFCollection >(
        grid_origin_,
        grid_dimensions_,
        gps_,
        soc_,
        ctc_
    );
}


template < typename CoordT >
inline const CoordT&
GFCollection< CoordT >::get_grid_origin() const
{
    return grid_origin_;
}


template < typename CoordT >
inline const GridPosition< CoordT >&
GFCollection< CoordT >::get_grid_dimensions() const
{
    return grid_dimensions_;
}


template < typename CoordT >
inline const std::unique_ptr< GridTargetPositionShifts< CoordT > >&
GFCollection< CoordT >::get_grid_shifts() const
{
    return gps_;
}

template < typename CoordT >
inline tileidx_t GFCollection< CoordT >::get_index(
    const GridPosition< CoordT >& pos
) const
{
    return position_to_index( pos, grid_dimensions_ );
}


template < typename CoordT >
inline CoordT GFCollection< CoordT >::shift_origin(
    const GridPosition< CoordT >& gpos,
    const GridPositionParity< CoordT >& gpa
) const
{
    return soc_->create_shifted_origin( grid_origin_, gpos, gpa );
}


template < typename CoordT >
inline std::unique_ptr< Tile< CoordT > > GFCollection< CoordT >::create_tile(
    const CoordT& tile_origin,
    const GridPositionParity< CoordT >& gpa
) const
{
    return ctc_->create_tile( tile_origin, gpa );
}
}


#endif
