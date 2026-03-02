/*
 *  masks2d.h
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

#ifndef MASKS2D_H
#define MASKS2D_H

#include "mask.h"
#include "mask_geometry.h"


namespace sapi
{
template < typename M2D >
class Mask2D : public Mask< Coord2D >
{
public:
    Mask2D() = delete;
    Mask2D( const Mask2D& ) = default;
    Mask2D( Mask2D&& ) = default;

    Mask2D( const space_t& radius )
        : Mask< Coord2D >( radius )
    {
    }

    Mask2D(
        const Coord2D& origin,
        const space_t& radius
    )
        : Mask< Coord2D >( origin, radius )
    {
    }

    bool overlap_with_tile_edges(
        const Tile< Coord2D >* const& tile
    ) const override;

    bool overlap_with_tile_edges(
        const Coord2D& coord,
        const Tile< Coord2D >* const& tile
    ) const override;

    std::unique_ptr< Mask< Coord2D > > clone() const override
    {
        return std::make_unique< M2D >( *dynamic_cast< const M2D* >( this ) );
    }
};


class CircularMask : public Mask2D< CircularMask >
{
public:
    CircularMask() = delete;
    CircularMask( const CircularMask& ) = default;
    CircularMask( CircularMask&& ) = default;

    CircularMask(
        const space_t& radius
    )
        : Mask2D( radius )
    {
    }

    CircularMask(
        const Coord2D& origin,
        const space_t& radius
    )
        : Mask2D( origin, radius )
    {
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& coord
        ) const override
    {
        return coord_in_circular_mask(
            origin_,
            radius2_,
            coord
        );
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& a,
            const Coord2D& b
        ) const override
    {
        return coord_in_circular_mask(
            offset_.has_value() ? a + offset_.value() : a,
            radius2_,
            b
        );
    }
};


class ParallelogramMask : public Mask2D< ParallelogramMask >
{
public:
    const Coord2D axial_vector0_;
    const Coord2D basis_vector0_;
    const Coord2D basis_vector1_;
    const space_t det_01_;

    ParallelogramMask() = delete;
    ParallelogramMask( const ParallelogramMask& ) = default;
    ParallelogramMask( ParallelogramMask&& ) = default;

    ParallelogramMask(
        const Coord2D& axial_vector0,
        const Coord2D& axial_vector1
    )
        : Mask2D(
            std::sqrt( std::fmax(
                vector_norm2( axial_vector0 ),
                vector_norm2( axial_vector1 )
            ) )
        )
        , axial_vector0_( axial_vector0 )
        , basis_vector0_( axial_vector0 - axial_vector1 )
        , basis_vector1_( axial_vector0 + axial_vector1 )
        , det_01_( coord_sum( vector_cross( basis_vector0_, basis_vector1_ ) ) )
    {
        assert( !almost_zero( det_01_ ) );
    }

    ParallelogramMask(
        const Coord2D& origin,
        const Coord2D& axial_vector0,
        const Coord2D& axial_vector1
    )
        : Mask2D(
            origin,
            std::sqrt( std::fmax(
                vector_norm2( axial_vector0 ),
                vector_norm2( axial_vector1 )
            ) )
        )
        , axial_vector0_( axial_vector0 )
        , basis_vector0_( axial_vector0 - axial_vector1 )
        , basis_vector1_( axial_vector0 + axial_vector1 )
        , det_01_( coord_sum( vector_cross( basis_vector0_, basis_vector1_ ) ) )
        , basis_vertex_( origin - axial_vector0 )
    {
        assert( !almost_zero( det_01_ ) );
    }

    void set_offset( Coord2D&& coord ) override
    {
        if ( has_origin_ )
        {
            origin_ = origin_ + coord;
            basis_vertex_ = basis_vertex_ + coord;
        }
        offset_.emplace( std::move( coord ) );
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& coord
        ) const
    {
        return coord_in_algebraic_mask(
            origin_,
            radius2_,
            basis_vertex_,
            basis_vector0_,
            basis_vector1_,
            det_01_,
            coord,
            false // triangular comparison
        );
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& a,
            const Coord2D& b
        ) const override
    {
        if ( offset_.has_value() )
        {
            const auto base = a + offset_.value();
            return coord_in_algebraic_mask(
                base,
                radius2_,
                base - axial_vector0_,
                basis_vector0_,
                basis_vector1_,
                det_01_,
                b,
                false // triangular comparison
            );
        }
        else
        {
            return coord_in_algebraic_mask(
                a,
                radius2_,
                a - axial_vector0_,
                basis_vector0_,
                basis_vector1_,
                det_01_,
                b,
                false // triangular comparison
            );
        }
    }

protected:
    Coord2D basis_vertex_;
};


class EllipticalMask : public Mask2D< EllipticalMask >
{
public:
    const std::optional< Coord2D > angular_offset_;
    const space_t semi_minor_axe2_;

    EllipticalMask() = delete;
    EllipticalMask( const EllipticalMask& ) = default;
    EllipticalMask( EllipticalMask&& ) = default;

    // Geometrically an ellipse does not have a radius,
    // however the semi major axe is the largest extent an ellipse can reach in one direction,
    // so it can still be used for early overlap rejection.
    EllipticalMask(
        const space_t& semi_major_axe,
        const space_t& semi_minor_axe,
        const space_t& rotation = 0
    )
        : Mask2D( semi_major_axe )
        , angular_offset_(
            almost_zero( rotation )
            ? std::optional< Coord2D >()
            : create_angular_offset( rotation )
        )
        , semi_minor_axe2_( squared( semi_minor_axe ) )
    {
        assert(
            !almost_zero( semi_minor_axe2_ ) &&
            !std::signbit( semi_minor_axe ) &&
            leq_test( semi_minor_axe, semi_major_axe )
        );
    }

    EllipticalMask(
        const Coord2D& origin,
        const space_t& semi_major_axe,
        const space_t& semi_minor_axe,
        const space_t& rotation = 0
    )
        : Mask2D( origin, semi_major_axe )
        , angular_offset_(
            almost_zero( rotation )
            ? std::optional< Coord2D >()
            : create_angular_offset( rotation )
        )
        , semi_minor_axe2_( squared( semi_minor_axe ) )
    {
        assert(
            !almost_zero( semi_minor_axe2_ ) &&
            !std::signbit( semi_minor_axe ) &&
            leq_test( semi_minor_axe, semi_major_axe )
        );
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& coord
        ) const
    {
        return coord_in_elliptical_mask(
            origin_,
            radius2_,
            semi_minor_axe2_,
            angular_offset_,
            coord
        );
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& a,
            const Coord2D& b
        ) const override
    {
        return coord_in_elliptical_mask(
            offset_.has_value() ? a + offset_.value() : a,
            radius2_,
            semi_minor_axe2_,
            angular_offset_,
            b
        );
    }
};


class TriangularMask : public Mask2D< ParallelogramMask >
{
public:
    const Coord2D axial_vector0_;
    const Coord2D axial_vector1_;
    const space_t det_01_;

    TriangularMask() = delete;
    TriangularMask( const TriangularMask& ) = default;
    TriangularMask( TriangularMask&& ) = default;

    TriangularMask(
        const Coord2D& axial_vector0,
        const Coord2D& axial_vector1
    )
        : Mask2D(
            std::sqrt( std::fmax(
                vector_norm2( axial_vector0 ),
                vector_norm2( axial_vector1 )
            ) )
        )
        , axial_vector0_( axial_vector0 )
        , axial_vector1_( axial_vector1 )
        , det_01_( coord_sum( vector_cross( axial_vector0_, axial_vector1_ ) ) )
    {
        assert( !almost_zero( det_01_ ) );
    }

    TriangularMask(
        const Coord2D& origin,
        const Coord2D& axial_vector0,
        const Coord2D& axial_vector1
    )
        : Mask2D(
            origin,
            std::sqrt( std::fmax(
                vector_norm2( axial_vector0 ),
                vector_norm2( axial_vector1 )
            ) )
        )
        , axial_vector0_( axial_vector0 )
        , axial_vector1_( axial_vector1 )
        , det_01_( coord_sum( vector_cross( axial_vector0_, axial_vector1_ ) ) )
    {
        assert( !almost_zero( det_01_ ) );
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& coord
        ) const
    {
        return coord_in_algebraic_mask(
            origin_,
            radius2_,
            origin_,
            axial_vector0_,
            axial_vector1_,
            det_01_,
            coord,
            true // triangular comparison
        );
    }

    std::optional< Displacement< Coord2D > >
        coord_in_mask(
            const Coord2D& a,
            const Coord2D& b
        ) const override
    {
        if ( offset_.has_value() )
        {
            const auto base = a + offset_.value();
            return coord_in_algebraic_mask(
                base,
                radius2_,
                base,
                axial_vector0_,
                axial_vector1_,
                det_01_,
                b,
                true // triangular comparison
            );
        }
        else
        {
            return coord_in_algebraic_mask(
                a,
                radius2_,
                a,
                axial_vector0_,
                axial_vector1_,
                det_01_,
                b,
                true // triangular comparison
            );
        }
    }
};


template < typename M2D >
bool Mask2D< M2D >::overlap_with_tile_edges(
    const Tile< Coord2D >* const& tile
) const
{
    bool res = false;
    assert( tile->get_vertices().size() < std::numeric_limits< vertidx_t >::max() );
    const auto vertex_count = static_cast< vertidx_t >( tile->get_vertices().size() );
    const auto tile_vertices = tile->get_vertices().data();
    for ( vertidx_t vix = 0; vix < vertex_count; ++vix )
    {
        res = coord_in_mask(
            projection_coord(
                origin_,
                tile_vertices[ vix ],
                tile_vertices[ ( vix + 1 ) % vertex_count ],
                true
            )
        ).has_value();
        if ( res ) break;
    }

    return res;
}


template < typename M2D >
bool Mask2D< M2D >::overlap_with_tile_edges(
    const Coord2D& coord,
    const Tile< Coord2D >* const& tile
) const
{
    bool res = false;
    assert( tile->get_vertices().size() < std::numeric_limits< vertidx_t >::max() );
    const auto vertex_count = static_cast< vertidx_t >( tile->get_vertices().size() );
    const auto tile_vertices = tile->get_vertices().data();
    for ( vertidx_t vix = 0; vix < vertex_count; ++vix )
    {
        res = coord_in_mask(
            coord,
            projection_coord(
                coord,
                tile_vertices[ vix ],
                tile_vertices[ ( vix + 1 ) % vertex_count ],
                true
            )
        ).has_value();
        if ( res ) break;
    }

    return res;
}
}


#endif
