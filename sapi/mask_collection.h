#ifndef MASK_COLLECTION_H
#define MASK_COLLECTION_H

#include "mask.h"
#include "creator_registry.h"


namespace sapi
{
template < typename CoordT >
class MaskCollection : public Clonable< MaskCollection< CoordT > >
{
public:
    MaskCollection() = delete;
    MaskCollection( const MaskCollection& ) = delete;
    MaskCollection( MaskCollection&& );

    MaskCollection(
        const std::unique_ptr< Mask< CoordT > >& blueprint,
        const std::unique_ptr< Mask< CoordT > >& source_mask,
        const std::unique_ptr< Mask< CoordT > >& target_mask
    );

    MaskCollection(
        const std::string& blueprint_name,
        const std::vector< space_t >& blueprint_params,
        const std::vector< space_t >& blueprint_offset,
        const std::string& source_mask_name,
        const std::vector< space_t >& source_mask_params,
        const std::vector< space_t >& source_mask_offset,
        const std::string& target_mask_name,
        const std::vector< space_t >& target_mask_params,
        const std::vector< space_t >& target_mask_offset,
        const CreatorRegistry< Mask< CoordT > >& mc_registry
    );

    MaskCollection& operator=( MaskCollection&& );

    std::unique_ptr< MaskCollection > clone() const;

    bool has_blueprint() const;
    bool has_source_mask() const;
    bool has_target_mask() const;

    std::optional< Displacement< CoordT > >
        blueprint_overlap(
            const CoordT& a, const CoordT& b
        ) const;

    bool blueprint_overlap(
        const Tile< CoordT >* const& a,
        const Tile< CoordT >* const& b
    ) const;

    bool blueprint_overlap(
        const CoordT& a,
        const Tile< CoordT >* const& b
    ) const;

    std::optional< Displacement< CoordT > >
        source_overlap( const CoordT& c ) const;

    bool source_overlap( const Tile< CoordT >* const& t ) const;

    std::forward_list< const Tile< CoordT >* >
        source_overlapping_leafs( const Tile< CoordT >* const& t ) const;

    std::optional< Displacement< CoordT > >
        target_overlap( const CoordT& c ) const;

    bool target_overlap( const Tile< CoordT >* const& t ) const;

    std::forward_list< const Tile< CoordT >* >
        target_overlapping_leafs( const Tile< CoordT >* const& t ) const;

protected:
    std::unique_ptr< Mask< CoordT > >  blueprint_;
    std::unique_ptr< Mask< CoordT > > source_mask_;
    std::unique_ptr< Mask< CoordT > > target_mask_;
};


template < typename CoordT >
MaskCollection< CoordT >::MaskCollection( MaskCollection< CoordT >&& mc )
{
    blueprint_ = std::move( mc.blueprint_ );
    source_mask_ = std::move( mc.source_mask_ );
    target_mask_ = std::move( mc.target_mask_ );
}


template < typename CoordT >
MaskCollection< CoordT >::MaskCollection(
    const std::unique_ptr< Mask< CoordT > >& blueprint,
    const std::unique_ptr< Mask< CoordT > >& source_mask,
    const std::unique_ptr< Mask< CoordT > >& target_mask
)
{
    if ( blueprint )
        blueprint_ = blueprint->clone();
    if ( source_mask )
        source_mask_ = source_mask->clone();
    if ( target_mask )
        target_mask_ = target_mask->clone();
}


template < typename CoordT >
MaskCollection< CoordT >::MaskCollection(
    const std::string& blueprint_name,
    const std::vector< space_t >& blueprint_params,
    const std::vector< space_t >& blueprint_offset,
    const std::string& source_mask_name,
    const std::vector< space_t >& source_mask_params,
    const std::vector< space_t >& source_mask_offset,
    const std::string& target_mask_name,
    const std::vector< space_t >& target_mask_params,
    const std::vector< space_t >& target_mask_offset,
    const CreatorRegistry< Mask< CoordT > >& mc_registry
)
{
    if ( !blueprint_name.empty() )
        blueprint_ = mc_registry.get_creator( blueprint_name )->create(
            blueprint_params, blueprint_offset
        );
    if ( !source_mask_name.empty() )
        source_mask_ = mc_registry.get_creator( source_mask_name )->create(
            source_mask_params, source_mask_offset
        );
    if ( !target_mask_name.empty() )
        target_mask_ = mc_registry.get_creator( target_mask_name )->create(
            target_mask_params, target_mask_offset
        );
}


template < typename CoordT >
MaskCollection< CoordT >& MaskCollection< CoordT >::operator=( MaskCollection&& mc )
{
    blueprint_ = std::move( mc.blueprint_ );
    source_mask_ = std::move( mc.source_mask_ );
    target_mask_ = std::move( mc.target_mask_ );
    return *this;
}


template < typename CoordT >
inline std::unique_ptr< MaskCollection< CoordT > >
MaskCollection< CoordT >::clone() const
{
    return std::make_unique< MaskCollection >(
        blueprint_,
        source_mask_,
        target_mask_
    );
}


template < typename CoordT >
inline bool MaskCollection< CoordT >::has_blueprint() const
{
    return bool( blueprint_ ) && !blueprint_->has_origin_;
}


template < typename CoordT >
inline bool MaskCollection< CoordT >::has_source_mask() const
{
    return bool( source_mask_ ) && source_mask_->has_origin_;
}


template < typename CoordT >
inline bool MaskCollection< CoordT >::has_target_mask() const
{
    return bool( target_mask_ ) && target_mask_->has_origin_;
}


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
MaskCollection< CoordT >::blueprint_overlap(
    const CoordT& a, const CoordT& b
) const
{
    assert( blueprint_ );
    return blueprint_->coord_in_mask( a, b );
}


template < typename CoordT >
inline bool MaskCollection< CoordT >::blueprint_overlap(
    const Tile< CoordT >* const& a,
    const Tile< CoordT >* const& b
) const
{
    assert( blueprint_ );
    return blueprint_->tiles_within_mask_range( a, b );
}


template < typename CoordT >
inline bool MaskCollection< CoordT >::blueprint_overlap(
    const CoordT& a,
    const Tile< CoordT >* const& b
) const
{
    assert( blueprint_ );
    return blueprint_->overlap_with_tile_edges( a, b );
}


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
MaskCollection< CoordT >::source_overlap(
    const CoordT& c
) const
{
    assert( source_mask_ );
    return source_mask_->coord_in_mask( c );
}


template < typename CoordT >
inline bool MaskCollection< CoordT >::source_overlap(
    const Tile< CoordT >* const& t
) const
{
    assert( source_mask_ );
    return source_mask_->tile_overlap(
        t
    ) != OverlapLevel::NONE;
}


template < typename CoordT >
inline std::forward_list< const Tile< CoordT >* >
MaskCollection< CoordT >::source_overlapping_leafs(
    const Tile< CoordT >* const& t
) const
{
    assert( source_mask_ );
    return source_mask_->get_overlapping_leaf_sub_tiles( t );
}


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
MaskCollection< CoordT >::target_overlap(
    const CoordT& c
) const
{
    assert( target_mask_ );
    return target_mask_->coord_in_mask( c );
}


template < typename CoordT >
inline bool MaskCollection< CoordT >::target_overlap(
    const Tile< CoordT >* const& t
) const
{
    assert( target_mask_ );
    return target_mask_->tile_overlap(
        t
    ) != OverlapLevel::NONE;
}


template < typename CoordT >
inline std::forward_list< const Tile< CoordT >* >
MaskCollection< CoordT >::target_overlapping_leafs(
    const Tile< CoordT >* const& t
) const
{
    assert( target_mask_ );
    return target_mask_->get_overlapping_leaf_sub_tiles( t );
}


template < typename CoordT >
inline bool no_overlap(
    const Tile< CoordT >* const& tile,
    const MaskCollection< CoordT >* const& mask_collection,
    const bool& apply_on_target
)
{
    return apply_on_target
        ? mask_collection->has_target_mask()
        ? !mask_collection->target_overlap( tile )
        : false
        : mask_collection->has_source_mask()
        ? !mask_collection->source_overlap( tile )
        : false;
}


template < typename CoordT >
inline bool no_overlap(
    const CoordT& coord,
    const MaskCollection< CoordT >* const& mask_collection,
    const bool& apply_on_target
)
{
    return apply_on_target
        ? mask_collection->has_target_mask()
        ? !mask_collection->target_overlap( coord ).has_value()
        : false
        : mask_collection->has_source_mask()
        ? !mask_collection->source_overlap( coord ).has_value()
        : false;
}


template < typename CoordT >
inline bool no_overlap(
    const CoordT& coord,
    const Tile< CoordT >* const& tile,
    const MaskCollection< CoordT >* const& mask_collection,
    const bool& apply_on_target
)
{
    return mask_collection->has_blueprint()
        ? !mask_collection->blueprint_overlap( coord, tile )
        : false;
}
}


#endif
