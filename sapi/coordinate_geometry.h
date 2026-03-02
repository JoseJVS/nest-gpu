/*
 *  coordinate_geometry.h
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

#ifndef COORDINATE_GEOMETRY_H
#define COORDINATE_GEOMETRY_H

#include <array>
#include <vector>
#include <cassert>

#include "coordinates.h"


namespace sapi
{
inline space_t coord_sum( const Coord2D& coord )
{
    return compensated_sum( coord.x_, coord.y_ );
}


inline space_t coord_sum( const Coord3D& coord )
{
    return compensated_sum( coord.x_, coord.y_, coord.z_ );
}


template < typename CoordT >
inline space_t vector_dot( const CoordT& coordA, const CoordT& coordB )
{
    return coord_sum( coordA * coordB );
}


inline Coord2D vector_cross( const Coord2D& coordA, const Coord2D& coordB )
{
    return Coord2D(
        coordA.x_ * coordB.y_,
        -( coordA.y_ * coordB.x_ )
    ).sanitize();
}


inline Coord3D vector_cross( const Coord3D& coordA, const Coord3D& coordB )
{
    return Coord3D(
        coord_sum(
            vector_cross(
                Coord2D( coordA.y_, coordB.y_ ),
                Coord2D( coordA.z_, coordB.z_ )
            )
        ),
        -coord_sum(
            vector_cross(
                Coord2D( coordA.x_, coordB.x_ ),
                Coord2D( coordA.z_, coordB.z_ )
            )
        ),
        coord_sum(
            vector_cross(
                Coord2D( coordA.x_, coordB.x_ ),
                Coord2D( coordA.y_, coordB.y_ )
            )
        )
    );
}


template < typename CoordT >
inline space_t vector_norm2( const CoordT& coord )
{
    return coord_sum( squared( coord ) );
}


template < typename CoordT >
inline space_t vector_norm( const CoordT& coord )
{
    return std::sqrt( vector_norm2( coord ) );
}


template < typename CoordT >
inline CoordT normalize_vector( const CoordT& coord )
{
    return coord.is_null()
        ? CoordT()
        : coord / vector_norm( coord );
}


template < typename CoordT >
inline space_t distance2( const CoordT& coordA, const CoordT& coordB )
{
    return vector_norm2( ( coordA - coordB ) );
}


template < typename CoordT >
inline space_t distance( const CoordT& coordA, const CoordT& coordB )
{
    return std::sqrt( distance2( coordA, coordB ) );
}


template < typename CoordT >
inline CoordT midpoint( const CoordT& coordA, const CoordT& coordB )
{
    return ( coordA + coordB ) / 2;
}


template < typename CoordT >
space_t projection_coeff(
    const CoordT& vectorBA,
    const CoordT& vectorBC,
    const bool& clamp
)
{
    // We project BA onto BC using orthogonal vector projection.
    const space_t dot_BA_BC = vector_dot( vectorBA, vectorBC );
    const space_t n2BC = vector_norm2( vectorBC );

    const bool n2_BC_az = almost_zero( n2BC );
    const bool dot_BA_BC_az = almost_zero( dot_BA_BC );

    assert( !n2_BC_az || dot_BA_BC_az );

    if ( n2_BC_az || dot_BA_BC_az ) return 0;

    if ( almost_equal( dot_BA_BC, n2BC ) ) return 1;

    // Safe to divide at this point
    return clamp
        ? std::fmin< space_t >(
            std::fmax< space_t >(
                clamp_epsilon_0( dot_BA_BC / n2BC ), 0 ), 1 )
        : clamp_epsilon_0( dot_BA_BC / n2BC );
}


template < typename CoordT >
inline CoordT projection_vector(
    const CoordT& vectorBA,
    const CoordT& vectorBC,
    const bool& clamp
)
{
    // We project BA onto BC using orthogonal vector projection.
    return vectorBC * projection_coeff< CoordT >( vectorBA, vectorBC, clamp );
}


template < typename CoordT >
inline CoordT projection_coord(
    const CoordT& coordA,
    const CoordT& coordB,
    const CoordT& coordC,
    const bool& clamp
)
{
    // We project A onto BC using orthogonal vector projection.
    return coordB + projection_vector< CoordT >( coordA - coordB, coordC - coordB, clamp );
}


inline Coord2D create_angular_offset( const space_t& angle )
{
    const space_t rad_angle = radians( angle );
    return Coord2D(
        std::cos( rad_angle ),
        std::sin( rad_angle )
    ).sanitize();
}


inline Coord2D rotate_displacement(
    const Coord2D& displacement,
    const Coord2D& angular_coord
)
{
    // With X = x * cos @ + y * sin @
    // and  Y = -x * sin @ + y * cos @ 
    return Coord2D(
        coord_sum( angular_coord * displacement ),
        coord_sum( vector_cross( angular_coord, displacement ) )
    );
}


template < typename CoordT >
bool algebraic_projection_comparison(
    const CoordT& vectorBA,
    const CoordT& vectorBC,
    const CoordT& vectorBD,
    const space_t& det_BCD,
    const bool& triangular_comparison
)
{
    // Solving comparison as equation system form
    // with pre-computed determinant
    const space_t cross_ABC = coord_sum(
        vector_cross( vectorBC, vectorBA )
    );
    const space_t cross_ABD = coord_sum(
        vector_cross( vectorBA, vectorBD )
    );

    const bool cross_ABC_az = almost_zero( cross_ABC );
    const bool cross_ABD_az = almost_zero( cross_ABD );

    if ( cross_ABC_az && cross_ABD_az )
        return true;

    const bool cross_ABC_aeq_det_BCD = almost_equal( cross_ABC, det_BCD );
    const bool cross_ABD_aeq_det_BCD = almost_equal( cross_ABD, det_BCD );

    if ( cross_ABC_aeq_det_BCD && cross_ABD_aeq_det_BCD )
        return !triangular_comparison;

    if ( cross_ABC_aeq_det_BCD && triangular_comparison )
        return cross_ABD_az;

    if ( cross_ABD_aeq_det_BCD && triangular_comparison )
        return cross_ABC_az;

    const space_t coeffABC = cross_ABC_az
        ? 0
        : cross_ABC_aeq_det_BCD
        ? 1
        : clamp_epsilon_0( cross_ABC / det_BCD );

    if ( !interval_test( 0, coeffABC, 1 ) )
        return false;

    const space_t coeffABD = cross_ABD_az
        ? 0
        : cross_ABD_aeq_det_BCD
        ? 1
        : clamp_epsilon_0( cross_ABD / det_BCD );

    if ( !interval_test( 0, coeffABD, 1 ) )
        return false;

    return leq_test(
        compensated_sum( coeffABC, coeffABD ),
        2 - static_cast< int >( triangular_comparison )
    );
}


template < typename CoordT >
vertidx_t sort_vertices(
    std::array< CoordT, 3>&& coord_in,
    std::array< CoordT, 3 >& coord_out,
    std::array< CoordT, 3 >& vectors,
    std::array< space_t, 3 >& distances2
)
{
    vertidx_t A_vix = 0, B_vix = 0, C_vix = 0, vix_next = 0, equal_n2 = 0;
    space_t largest_n2 = 0, prev_n2 = 0;

    CoordT indexed_vectors[ 3 ];
    bool almost_equal_n2[ 3 ] = { false };
    space_t edge_n2[ 3 ] = { 0 };
    for ( vertidx_t vix = 0; vix < 3; ++vix )
    {
        vix_next = ( vix + 1 ) % 3;
        indexed_vectors[ vix ] = coord_in[ vix_next ] - coord_in[ vix ];
        edge_n2[ vix ] = vector_norm2( indexed_vectors[ vix ] );
        assert( !almost_zero( edge_n2[ vix ] ) );

        if ( vix == 0 )
        {
            A_vix = vix;
            B_vix = vix_next;
            largest_n2 = edge_n2[ vix ];
        }
        else
        {
            if ( almost_equal( prev_n2, edge_n2[ vix ] ) )
            {
                almost_equal_n2[ vix - 1 ] = true;
                ++equal_n2;
            }
            else if ( leq_test( largest_n2, edge_n2[ vix ] ) )
            {
                A_vix = vix;
                B_vix = vix_next;
                largest_n2 = edge_n2[ vix ];
            }
        }

        prev_n2 = edge_n2[ vix ];
    }
    if ( almost_equal( prev_n2, edge_n2[ 0 ] ) )
    {
        almost_equal_n2[ 2 ] = true;
        ++equal_n2;
    }
    C_vix = ( B_vix + 1 ) % 3;

    // Due to commutative comparison equal lengths == 2 implies equal lengths == 3
    if ( equal_n2 == 2 )
        equal_n2 = 3;

    // A triangle either starts as a right triangle from a square
    // i.e. isosceles right triangle or an equilateral triangle.
    // When spliting an isosceles right triangle, symmetry is preserved
    // and resulting triangles are isosceles right triangles too.
    // When splitting an equilateral triangle resulting triangles
    // are two right triangles with no equal lengths.
    // For right triangles with no equal lengths we return
    // the hypotenuse vertices and apex vertex in that order.
    // For an equilateral triangle any combination is fine.
    if ( equal_n2 != 1 )
    {
        coord_out[ 0 ] = std::move( coord_in[ A_vix ] );
        coord_out[ 1 ] = std::move( coord_in[ B_vix ] );
        coord_out[ 2 ] = std::move( coord_in[ C_vix ] );

        vectors[ 0 ] = std::move( indexed_vectors[ A_vix ] );
        vectors[ 1 ] = std::move( indexed_vectors[ B_vix ] );
        vectors[ 2 ] = std::move( indexed_vectors[ C_vix ] );

        distances2[ 0 ] = std::move( edge_n2[ A_vix ] );
        distances2[ 1 ] = std::move( edge_n2[ B_vix ] );
        distances2[ 2 ] = std::move( edge_n2[ C_vix ] );
    }
    // For an isosceles triangle (right or not) we return the base vertices
    // and apex vertex in that order.
    else
    {
        // Get end vertex of base edge of isosceles triangle
        vertidx_t vix_base_end = 0;
        for ( ; vix_base_end < 3; ++vix_base_end )
            if ( almost_equal_n2[ vix_base_end ] )
                break;
        const vertidx_t vix_base_start = ( vix_base_end + 2 ) % 3;
        const vertidx_t vix_apex = ( vix_base_end + 1 ) % 3;

        coord_out[ 0 ] = std::move( coord_in[ vix_base_start ] );
        coord_out[ 1 ] = std::move( coord_in[ vix_base_end ] );
        coord_out[ 2 ] = std::move( coord_in[ vix_apex ] );

        vectors[ 0 ] = std::move( indexed_vectors[ vix_base_start ] );
        vectors[ 1 ] = std::move( indexed_vectors[ vix_base_end ] );
        vectors[ 2 ] = std::move( indexed_vectors[ vix_apex ] );

        distances2[ 0 ] = std::move( edge_n2[ vix_base_start ] );
        distances2[ 1 ] = std::move( edge_n2[ vix_base_end ] );
        distances2[ 2 ] = std::move( edge_n2[ vix_apex ] );
    }

    return equal_n2;
}


template < typename CoordT >
CircumscribedRadius< CoordT > compute_c_radius(
    const CoordT& coordA,
    const CoordT& coordB,
    const CoordT& coordC
)
{
    CoordT origin;
    space_t radius2;

    std::array< CoordT, 3 > sorted_coords;
    std::array< CoordT, 3 > sorted_vectors;
    std::array< space_t, 3 > sorted_distances2;
    const auto equal_lengths = sort_vertices(
        { coordA, coordB, coordC },
        sorted_coords,
        sorted_vectors,
        sorted_distances2
    );

    const CoordT base_vertices_add( sorted_coords[ 0 ] + sorted_coords[ 1 ] );

    if ( equal_lengths == 3 )
    {
        // Implies equilateral triangle, circumscribed center is
        // a scaled orthogonal projection of apex vertex onto base edge
        // here apex vertex and base edge are an arbitrary choice

        // With side length a, radius r is equal to a / sqrt( 3 )
        radius2 = sorted_distances2[ 0 ] / 3;

        // With side length a, height h is equal to sqrt( 3 ) * a / 2
        // projection coefficient of apex vertex onto base edge is r / h == 2 / 3
        // Using vector combining formula of apex + ( midlepoint of base edge - apex ) * coefficient
        // we reduce to apex * ( 1 - coefficient ) + ( base edge * coefficient / 2 )
        // with coefficient == 2 / 3 we have apex / 3 + base edge / 3
        origin = ( sorted_coords[ 2 ] / 3 ) + ( base_vertices_add / 3 );
    }
    else
    {
        if (
            almost_zero(
                vector_dot( sorted_vectors[ 1 ], sorted_vectors[ 2 ] )
            )
            )
        {
            // Right triangle case, circumscribed center is the midpoint of the hypotenuse
            // hypotenuse == base and radius == | base | / 2
            origin = base_vertices_add / 2;
            radius2 = sorted_distances2[ 0 ] / 4;
        }
        else if ( equal_lengths == 1 )
        {
            // Plain isosceles case, circumscribed center is
            // a scaled orthogonal projection of apex vertex onto base edge

            // With base length b and side length a
            // height h is equal to sqrt( a2 - ( b2 / 4 ) )
            // => 4 * h2 = 4 * a2 - b2
            const space_t height2_4 = compensated_sum(
                4 * sorted_distances2[ 2 ], -sorted_distances2[ 0 ]
            );

            // With base height h and side length a
            // radius r is equal to a2 / ( 2 * h )
            //  => radius2 = a4 / ( 4 * h2 )
            radius2 = squared( sorted_distances2[ 2 ] ) / height2_4;

            // Projection coefficient of apex vertex onto base edge is r / h
            const space_t projection_coefficient = sorted_distances2[ 2 ] / (
                height2_4 / 2
                );

            origin = sorted_coords[ 2 ] * compensated_sum( 1., -projection_coefficient )
                + base_vertices_add * ( projection_coefficient / 2 );
        }
        else
        {
            // Compute 2D circumscribed center via equation system
            // to find equidistant point to all vertices
            // solved using Cramer's rule
            // Compute 3D circumscribed center by projecting 3D
            // coordinates onto 2D plane, then using 2D equation
            // system, and finally projecting back to 3D space
            origin = circumscribed_center( coordA, coordB, coordC );
            radius2 = distance2( origin, coordA );
        }
    }

    return CircumscribedRadius< CoordT >(
        std::move( origin ),
        std::move( radius2 )
    );
}


template < typename CoordT >
constexpr void update_min( CoordT& min, const CoordT& value )
{
    min.x_ = value.x_ < min.x_ ? value.x_ : min.x_;
    min.y_ = value.y_ < min.y_ ? value.y_ : min.y_;

    if constexpr ( std::is_same_v< rcvref< CoordT >, Coord3D > )
        min.z_ = value.z_ < min.z_ ? value.z_ : min.z_;
}


template < typename CoordT >
constexpr void update_max( CoordT& max, const CoordT& value )
{
    max.x_ = max.x_ < value.x_ ? value.x_ : max.x_;
    max.y_ = max.y_ < value.y_ ? value.y_ : max.y_;

    if constexpr ( std::is_same_v< rcvref< CoordT >, Coord3D > )
        max.z_ = max.z_ < value.z_ ? value.z_ : max.z_;
}


template < typename CoordT >
inline std::pair< CoordT, CoordT >
minmax_coords()
{

    CoordT min, max;
    min.x_ = std::numeric_limits< space_t >::max();
    min.y_ = std::numeric_limits< space_t >::max();
    max.x_ = std::numeric_limits< space_t >::lowest();
    max.y_ = std::numeric_limits< space_t >::lowest();

    if constexpr ( std::is_same_v< rcvref< CoordT >, Coord3D > )
    {
        min.z_ = std::numeric_limits< space_t >::max();
        max.z_ = std::numeric_limits< space_t >::lowest();
    }

    return std::make_pair(
        std::move( min ),
        std::move( max )
    );
}


template < typename CoordT >
std::pair< CoordT, CoordT >
minmax_vertices(
    const std::vector< CoordT >& coord_vec
)
{
    assert( 1 < coord_vec.size() );

    auto [min, max] = minmax_coords< CoordT >();

    for ( const auto& coord : coord_vec )
    {
        update_min( min, coord );
        update_max( max, coord );
    }

    return std::make_pair(
        std::move( min ),
        std::move( max )
    );
}


inline Coord2D circumscribed_center(
    const Coord2D& coordA,
    const Coord2D& coordB,
    const Coord2D& coordC
)
{
    // circumscribed center ( x, y ) of triangle with vertices A, B, C is such that
    //   (x - xA)^2 + (y - yA)^2
    // = (x - xB)^2 + (y - yB)^2
    // = (x - xC)^2 + (y - yC)^2
    // < = >
    //   x^2 -2xxA + xA^2 + y^2 -2yyA + yA^2
    // = x^2 -2xxB + xB^2 + y^2 -2yyB + yB^2
    // = x^2 -2xxC + xC^2 + y^2 -2yyC + yC^2
    // < = >
    //   -2xxA + xA^2 -2yyA + yA^2
    // = -2xxB + xB^2 -2yyB + yB^2
    // = -2xxC + xC^2 -2yyC + yC^2
    // let D = -2xxA + xA^2 -2yyA + yA^2
    // D = -2xxA + xA^2 -2yyA + yA^2
    // D = -2xxB + xB^2 -2yyB + yB^2
    // D = -2xxC + xC^2 -2yyC + yC^2
    // < = >
    // D + 2xxA + 2yyA = xA^2 + yA^2
    // D + 2xxB + 2yyB = xB^2 + yB^2
    // D + 2xxC + 2yyC = xC^2 + yC^2
    // as a matrix equation system
    // | 1 2xA 2yA |   | D |   | xA^2 + yA^2 |
    // | 1 2xB 2yB | * | x | = | xB^2 + yB^2 |
    // | 1 2xC 2yC |   | y |   | xC^2 + yC^2 |
    // which is solved with Cramer's rule by
    //           | xA^2 + yA^2 2xA 2yA |          | 1 2xA 2yA |
    // D = det ( | xB^2 + yB^2 2xB 2yB | ) / det( | 1 2xB 2yB | )
    //           | xC^2 + yC^2 2xC 2yC |          | 1 2xC 2yC |
    //           | 1 xA^2 + yA^2 2yA |          | 1 2xA 2yA |
    // x = det ( | 1 xB^2 + yB^2 2yB | ) / det( | 1 2xB 2yB | )
    //           | 1 xC^2 + yC^2 2yC |          | 1 2xC 2yC |
    //           | 1 2xA xA^2 + yA^2 |          | 1 2xA 2yA |
    // y = det ( | 1 2xB xB^2 + yB^2 | ) / det( | 1 2xB 2yB | )
    //           | 1 2xC xC^2 + yC^2 |          | 1 2xC 2yC |

    const Coord2D twoA = coordA * 2;
    const Coord2D twoB = coordB * 2;
    const Coord2D twoC = coordC * 2;

    const space_t det_denominator = determinant3x3(
        1, twoA.x_, twoA.y_,
        1, twoB.x_, twoB.y_,
        1, twoC.x_, twoC.y_
    );

    const bool dd_az = almost_zero( det_denominator );

    const space_t n2A = vector_norm2( coordA );
    const space_t n2B = vector_norm2( coordB );
    const space_t n2C = vector_norm2( coordC );

    const space_t det_x_numerator = determinant3x3(
        1, n2A, twoA.y_,
        1, n2B, twoB.y_,
        1, n2C, twoC.y_
    );

    const bool dxn_az = almost_zero( det_x_numerator );

    const space_t det_y_numerator = determinant3x3(
        1, twoA.x_, n2A,
        1, twoB.x_, n2B,
        1, twoC.x_, n2C
    );

    const bool dyn_az = almost_zero( det_y_numerator );

    assert( !dd_az || ( dxn_az && dyn_az ) );

    if ( dd_az || ( dxn_az && dyn_az ) )
        return Coord2D();

    const space_t coord_x = dxn_az
        ? 0
        : almost_equal( det_x_numerator, det_denominator )
        ? 1
        : clamp_epsilon_0( det_x_numerator / det_denominator );

    const space_t coord_y = dyn_az
        ? 0
        : almost_equal( det_y_numerator, det_denominator )
        ? 1
        : clamp_epsilon_0( det_y_numerator / det_denominator );

    return Coord2D( std::move( coord_x ), std::move( coord_y ) );
}


inline Coord3D circumscribed_center(
    const Coord3D& coordA,
    const Coord3D& coordB,
    const Coord3D& coordC
)
{
    // Here we compute the 2D plane intersecting A, B, and C
    const Coord3D vBA = coordB - coordA;
    const Coord3D vCA = coordC - coordA;

    // A is now the origin of our plane
    // We now compute the basis vectors of the plane
    const Coord3D basis_1 = normalize_vector( vBA );
    assert( !basis_1.is_null() );

    // Gram-Schmidt process for orthonormal basis
    const Coord3D basis_2 = normalize_vector( vCA - projection_vector( vCA, basis_1, false ) );
    assert( !basis_2.is_null() );

    // Redefine A, B, and C as Coord2D in the plane and
    // compute the circumscribed center in 2D
    const Coord2D planar_circumscribed_center = circumscribed_center(
        Coord2D(), // A is the origin of the plane
        Coord2D(
            vector_dot( vBA, basis_1 ),
            vector_dot( vBA, basis_2 )
        ),
        Coord2D(
            vector_dot( vCA, basis_1 ),
            vector_dot( vCA, basis_2 )
        )
    );

    // project the center back to the 3D space
    const Coord3D x_projection = basis_1 * planar_circumscribed_center.x_;
    const Coord3D y_projection = basis_2 * planar_circumscribed_center.y_;
    return Coord3D(
        compensated_sum( coordA.x_, x_projection.x_, y_projection.x_ ),
        compensated_sum( coordA.y_, x_projection.y_, y_projection.y_ ),
        compensated_sum( coordA.z_, x_projection.z_, y_projection.z_ )
    );
}
}


#endif
