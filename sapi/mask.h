#ifndef MASK_H
#define MASK_H

#include "tile.h"
#include "type_erasure_helpers.h"


namespace sapi
{
enum OverlapLevel
{
    NONE,
    PARTIAL,
    FULL
};


template < typename CoordT >
class Mask : public Clonable< Mask < CoordT > >
{
public:
    const bool has_origin_;
    const space_t radius2_;

    Mask() = delete;
    Mask( const Mask& ) = default;
    Mask( Mask&& ) = default;

    Mask( const space_t& radius );

    Mask( const CoordT& origin, const space_t& radius );

    virtual void set_offset( CoordT&& offset );

    virtual std::optional< Displacement< CoordT > >
        coord_in_mask(
            const CoordT& coord
        ) const = 0;

    virtual std::optional< Displacement< CoordT > >
        coord_in_mask(
            const CoordT& a,
            const CoordT& b
        ) const = 0;

    virtual bool overlap_with_tile_edges(
        const Tile< CoordT >* const& tile
    ) const = 0;

    virtual bool overlap_with_tile_edges(
        const CoordT& coord,
        const Tile< CoordT >* const& tile
    ) const = 0;

    bool tiles_within_mask_range(
        const Tile< CoordT >* const& a,
        const Tile< CoordT >* const& b
    ) const;

    OverlapLevel tile_overlap(
        const Tile< CoordT >* const& tile,
        const std::optional< CoordT >& coord = {}
    ) const;

    std::forward_list< const Tile< CoordT >* >
        get_overlapping_leaf_sub_tiles(
            const Tile< CoordT >* const& tile,
            const std::optional< CoordT >& coord = {}
        ) const;

protected:
    CoordT origin_;
    std::optional< CoordT > offset_;
};


template < typename CoordT >
Mask< CoordT >::Mask( const space_t& radius )
    : has_origin_( false )
    , radius2_( squared( radius ) )
{
    assert( !almost_zero( radius2_ ) && !std::signbit( radius ) );
}


template < typename CoordT >
Mask< CoordT >::Mask( const CoordT& origin, const space_t& radius )
    : has_origin_( true )
    , radius2_( squared( radius ) )
    , origin_( origin )
{
    assert( !almost_zero( radius2_ ) && !std::signbit( radius ) );
}


template < typename CoordT >
inline void Mask< CoordT >::set_offset( CoordT&& coord )
{
    if ( has_origin_ )
        origin_ = origin_ + coord;
    offset_.emplace( std::move( coord ) );
}


template < typename CoordT >
bool Mask< CoordT >::tiles_within_mask_range(
    const Tile< CoordT >* const& a,
    const Tile< CoordT >* const& b
) const
{
    return coord_in_mask(
        a->project_point_to_surface(
            b->c_radius_.origin_
        ),
        b->project_point_to_surface(
            a->c_radius_.origin_
        )
    ).has_value();
}


template < typename CoordT >
OverlapLevel Mask< CoordT >::tile_overlap(
    const Tile< CoordT >* const& tile,
    const std::optional< CoordT >& coord
) const
{
    const auto target = coord.has_value() ? coord.value() : origin_;
    // Fast rejection check if mask origin is farther than sum of both radi
    if ( !tile->c_radius_.disp_in_radius( tile->c_radius_.origin_ - target, std::sqrt( radius2_ ) ) )
        return OverlapLevel::NONE;

    // Check if tile is fully in mask
    bool in_mask = true;
    vertidx_t vertices_in_mask = 0;
    auto tile_vit = tile->get_vertices().cbegin();
    const auto tile_vend = tile->get_vertices().cend();
    for ( ; tile_vit != tile_vend; ++tile_vit )
    {
        in_mask &= coord_in_mask(
            target,
            *tile_vit
        ).has_value();
        if ( !in_mask ) break; // Stop at first failure
        ++vertices_in_mask;
    }

    // If tile fully in mask
    if ( in_mask ) return OverlapLevel::FULL;

    // If at least one vertex is in mask
    if ( 0 < vertices_in_mask ) return OverlapLevel::PARTIAL;

    // If mask origin is in tile or vice versa
    if (
        tile->coord_in_tile( target ) ||
        coord_in_mask( target, tile->c_radius_.origin_ ).has_value()
        )
        return OverlapLevel::PARTIAL;

    // Continue with vertex check if possible
    // skipped if previous loop went over all vertices
    for ( ; tile_vit != tile_vend; ++tile_vit )
        if ( coord_in_mask(
            target,
            *tile_vit
        ).has_value() )
            return OverlapLevel::PARTIAL;

    // Check for mask area overlap with tile edges
    if ( overlap_with_tile_edges( target, tile ) )
        return OverlapLevel::PARTIAL;

    // Tile not in mask, no vertex in mask, mask not in tile, no edge overlap
    return OverlapLevel::NONE;
}


template < typename CoordT >
std::forward_list< const Tile< CoordT >* >
Mask< CoordT >::get_overlapping_leaf_sub_tiles(
    const Tile< CoordT >* const& tile,
    const std::optional< CoordT >& coord
) const
{
    std::forward_list< const Tile< CoordT >* > leaf_sub_tiles;
    std::forward_list< const Tile< CoordT >* > temps{ tile };
    do
    {
        auto current = temps.front();
        temps.pop_front();

        // Get level of overlap with tile
        switch ( tile_overlap( current, coord ) )
        {
            // If the tile is completely within the mask we insert all of
            // its leaf sub tiles ( defaults to the tile in question if no sub tiles available )
        case OverlapLevel::FULL:
            current->insert_leaf_sub_tiles( leaf_sub_tiles );
            break;

            // If there is only a partial overlap then we check into the sub-tiles
            // if there are no sub-tiles then we add the current tile to our result list
        case OverlapLevel::PARTIAL:
            if ( current->is_split() )
            {
                for ( const auto& st : current->get_sub_tiles() )
                    temps.emplace_front( st.get() );
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
