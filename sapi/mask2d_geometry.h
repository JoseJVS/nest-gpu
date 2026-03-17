/*
 *  mask2d_geometry.h
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

#ifndef MASK2D_GEOMETRY_H
#define MASK2D_GEOMETRY_H

#include "mask.h"
#include "coordinates.h"


namespace sapi
{
//Forward definition to link with coordinate_geometry.h
Coord2D create_angular_offset( const space_t& );
Coord2D rotate_displacement( const Coord2D&, const Coord2D& );
space_t coord_sum( const Coord2D& );
Coord2D vector_cross( const Coord2D&, const Coord2D& );
template < typename CoordT >
space_t vector_norm2( const CoordT& );
template < typename CoordT >
bool algebraic_projection_comparison(
    const CoordT&,
    const CoordT&,
    const CoordT&,
    const space_t&,
    const bool&
);


inline void initialize_circular_mask_helpers(
    space_t& radius2,
    const std::vector< space_t >& mask_params
)
{
    radius2 = squared( mask_params[ 0 ] );
    if ( std::signbit( mask_params[ 0 ] ) || almost_zero( radius2 ) )
        throw std::invalid_argument( "Invalid circular mask param" );
}


inline void initialize_elliptical_mask_helpers(
    space_t& radius2,
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< space_t >& mask_params
)
{
    const auto axis0 = mask_params[ 0 ];
    const auto axis1 = mask_params[ 1 ];

    if ( std::signbit( axis0 ) || std::signbit( axis1 ) )
        throw std::invalid_argument( "Invalid elliptical mask axes" );

    if ( std::isless( axis0, axis1 ) )
    {
        radius2 = squared( axis1 );
        helper_scalars.resize( 1, squared( axis0 ) );
    }
    else
    {
        radius2 = squared( axis0 );
        helper_scalars.resize( 1, squared( axis1 ) );
    }

    if ( almost_zero( radius2 ) || almost_zero( helper_scalars[ 0 ] ) )
        throw std::invalid_argument( "Invalid elliptical mask axes" );

    const space_t rotation = 2 < mask_params.size() ? mask_params[ 2 ] : 0;
    if ( !almost_zero( rotation ) )
        helper_vectors.resize( 1, create_angular_offset( rotation ) );
}


inline void initialize_parallelogram_mask_helpers(
    space_t& radius2,
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const Coord2D& origin,
    const std::vector< space_t >& mask_params
)
{
    auto it = mask_params.begin();
    auto axial_vector0 = Coord2D::copy_from_vec( it );
    auto axial_vector1 = Coord2D::copy_from_vec( it );

    radius2 = std::fmax(
        vector_norm2( axial_vector0 ),
        vector_norm2( axial_vector1 )
    );

    if ( almost_zero( radius2 ) )
        throw std::invalid_argument( "Invalid parallelogram mask vectors" );

    helper_vectors.reserve( 4 );
    helper_vectors.emplace_back( origin - axial_vector0 );
    helper_vectors.emplace_back( axial_vector0 - axial_vector1 );
    helper_vectors.emplace_back( axial_vector0 + axial_vector1 );
    helper_vectors.emplace_back( std::move( axial_vector0 ) );

    helper_scalars.resize( 1,
        coord_sum( vector_cross( helper_vectors[ 1 ], helper_vectors[ 2 ] ) )
    );

    if ( almost_zero( helper_scalars[ 0 ] ) )
        throw std::invalid_argument( "Invalid parallelogram mask vectors" );
}


inline void initialize_triangular_mask_helpers(
    space_t& radius2,
    std::vector< Coord2D >& helper_vectors,
    std::vector< space_t >& helper_scalars,
    const std::vector< space_t >& mask_params
)
{
    auto it = mask_params.begin();
    auto axial_vector0 = Coord2D::copy_from_vec( it );
    auto axial_vector1 = Coord2D::copy_from_vec( it );

    radius2 = std::fmax(
        vector_norm2( axial_vector0 ),
        vector_norm2( axial_vector1 )
    );

    if ( almost_zero( radius2 ) )
        throw std::invalid_argument( "Invalid triangular mask vectors" );

    helper_scalars.resize( 1,
        coord_sum( vector_cross( axial_vector0, axial_vector1 ) )
    );

    if ( almost_zero( helper_scalars[ 0 ] ) )
        throw std::invalid_argument( "Invalid triangular mask vectors" );

    helper_vectors.reserve( 2 );
    helper_vectors.emplace_back( std::move( axial_vector0 ) );
    helper_vectors.emplace_back( std::move( axial_vector1 ) );
}


inline std::optional< Displacement< Coord2D > >
    coord_in_circular_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& coord
    )
{
    Displacement< Coord2D > disp( coord - mask.origin_.value() );
    return leq_test( disp.distance2_, mask.radius2_ )
        ? std::make_optional( std::move( disp ) )
        : std::optional< Displacement< Coord2D > >();
}


inline std::optional< Displacement< Coord2D > >
    coord_in_circular_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& a,
        const Coord2D& b
    )
{
    Displacement< Coord2D > disp( b - ( mask.offset_.has_value() ? a + mask.offset_.value() : a ) );
    return leq_test( disp.distance2_, mask.radius2_ )
        ? std::make_optional( std::move( disp ) )
        : std::optional< Displacement< Coord2D > >();
}


inline bool rotated_coord_in_elliptical_mask(
    const Coord2D& displacement,
    const space_t& semi_major_axe2,
    const space_t& semi_minor_axe2
)
{
    const bool dx_az = almost_zero( displacement.x_ );
    const bool dy_az = almost_zero( displacement.y_ );

    if ( dx_az && dy_az )
        return true;

    if ( dx_az )
        return leq_test( squared( displacement.y_ ), semi_minor_axe2 );

    if ( dy_az )
        return leq_test( squared( displacement.x_ ), semi_major_axe2 );

    return leq_test(
        compensated_sum(
            squared( displacement.x_ ) / semi_major_axe2,
            squared( displacement.y_ ) / semi_minor_axe2
        ),
        1
    );
}


inline std::optional< Displacement< Coord2D > >
    coord_in_elliptical_mask(
        const Coord2D& mask_origin,
        const space_t& semi_major_axe2,
        const space_t& semi_minor_axe2,
        const std::vector< Coord2D >& helper_vectors,
        const Coord2D& coord
    )
{
    // Get displacement from mask origin
    Displacement< Coord2D > disp( coord - mask_origin );

    // Fast rejection method
    if ( !leq_test( disp.distance2_, semi_major_axe2 ) )
        return {};

    // Given ellipse with h, k center, a semi major, b semi minor, @ rotation
    // If x, y point is in ellipse then 
    // X^2 / a^2 + Y^2 / b^2 <= 1
    // With X = ( x - h ) * cos @ + ( y - k ) * sin @
    // and  Y = - ( x - h ) * sin @ + ( y - k ) * cos @ 
    return rotated_coord_in_elliptical_mask(
        helper_vectors.empty()
        ? disp.displacement_
        : rotate_displacement( disp.displacement_, helper_vectors[ 0 ] ),
        semi_major_axe2,
        semi_minor_axe2
    )
        ? std::make_optional( std::move( disp ) )
        : std::optional< Displacement< Coord2D > >();
}


inline std::optional< Displacement< Coord2D > >
    coord_in_elliptical_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& coord
    )
{
    return coord_in_elliptical_mask(
        mask.origin_.value(),
        mask.radius2_,
        mask.helper_scalars_[ 0 ],
        mask.helper_vectors_,
        coord
    );
}


inline std::optional< Displacement< Coord2D > >
    coord_in_elliptical_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& a,
        const Coord2D& b
    )
{
    return coord_in_elliptical_mask(
        mask.offset_.has_value() ? a + mask.offset_.value() : a,
        mask.radius2_,
        mask.helper_scalars_[ 0 ],
        mask.helper_vectors_,
        b
    );
}


inline std::optional< Displacement< Coord2D > >
    coord_in_algebraic_mask(
        const Coord2D& mask_origin,
        const space_t& circular_radius2,
        const Coord2D& basis_vertex,
        const Coord2D& basis_vector0,
        const Coord2D& basis_vector1,
        const space_t& det_01,
        const Coord2D& coord,
        const bool& triangular_comparison
    )
{
    // Get displacement from mask origin
    Displacement< Coord2D > disp( coord - mask_origin );

    // Fast rejection method
    if ( !leq_test( disp.distance2_, circular_radius2 ) )
        return {};

    // Get projection coefficients of coord onto parallelogram basis vectors
    return algebraic_projection_comparison(
        coord - basis_vertex,
        basis_vector0,
        basis_vector1,
        det_01,
        triangular_comparison
    )
        ? std::make_optional( std::move( disp ) )
        : std::optional< Displacement< Coord2D > >();
}


inline std::optional< Displacement< Coord2D > >
    coord_in_parallelogram_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& coord
    )
{
    return coord_in_algebraic_mask(
        mask.origin_.value(),
        mask.radius2_,
        mask.helper_vectors_[ 0 ],
        mask.helper_vectors_[ 1 ],
        mask.helper_vectors_[ 2 ],
        mask.helper_scalars_[ 0 ],
        coord,
        false // triangular comparison
    );
}


inline std::optional< Displacement< Coord2D > >
    coord_in_parallelogram_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& a,
        const Coord2D& b
    )
{
    if ( mask.offset_.has_value() )
    {
        const auto base = a + mask.offset_.value();
        return coord_in_algebraic_mask(
            base,
            mask.radius2_,
            base - mask.helper_vectors_[ 3 ],
            mask.helper_vectors_[ 1 ],
            mask.helper_vectors_[ 2 ],
            mask.helper_scalars_[ 0 ],
            b,
            false // triangular comparison
        );
    }
    else
    {
        return coord_in_algebraic_mask(
            a,
            mask.radius2_,
            a - mask.helper_vectors_[ 3 ],
            mask.helper_vectors_[ 1 ],
            mask.helper_vectors_[ 2 ],
            mask.helper_scalars_[ 0 ],
            b,
            false // triangular comparison
        );
    }
}


inline std::optional< Displacement< Coord2D > >
    coord_in_triangular_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& coord
    )
{
    return coord_in_algebraic_mask(
        mask.origin_.value(),
        mask.radius2_,
        mask.origin_.value(),
        mask.helper_vectors_[ 0 ],
        mask.helper_vectors_[ 1 ],
        mask.helper_scalars_[ 0 ],
        coord,
        true // triangular comparison
    );
}


inline std::optional< Displacement< Coord2D > >
    coord_in_triangular_mask(
        const Mask< Coord2D >& mask,
        const Coord2D& a,
        const Coord2D& b
    )
{
    if ( mask.offset_.has_value() )
    {
        const auto base = a + mask.offset_.value();
        return coord_in_algebraic_mask(
            base,
            mask.radius2_,
            base,
            mask.helper_vectors_[ 0 ],
            mask.helper_vectors_[ 1 ],
            mask.helper_scalars_[ 0 ],
            b,
            true // triangular comparison
        );
    }
    else
    {
        return coord_in_algebraic_mask(
            a,
            mask.radius2_,
            a,
            mask.helper_vectors_[ 0 ],
            mask.helper_vectors_[ 1 ],
            mask.helper_scalars_[ 0 ],
            b,
            true // triangular comparison
        );
    }
}
}


#endif
