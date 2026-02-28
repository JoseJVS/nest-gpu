#ifndef NUMERICS_H
#define NUMERICS_H

#include <limits>
#include <utility>
#include <type_traits>
#include <cmath>

#include "sapi_config.h"


namespace sapi
{
constexpr static const space_t SP_PI_180 = space_t( M_PI ) / space_t( 180 );
constexpr static const space_t SP_EPSILON = std::numeric_limits< space_t >::epsilon() * TOLERANCE;
constexpr static const space_t SP_RELATIVE_EPSILON = space_t( 1 ) / ( SP_EPSILON * ( 1L << RELATIVE_TOLERANCE ) );


template < typename T >
using rcvref = typename std::remove_cv_t<
    typename std::remove_reference_t< T >
>;


template < typename FROM, typename TO,
    typename std::enable_if_t< std::conjunction_v<
    std::is_floating_point< rcvref< FROM > >,
    std::is_floating_point< rcvref< TO > >
>, bool > = true >
constexpr TO safe_convert_f( const FROM& val )
{
    if constexpr ( std::is_same_v< rcvref< FROM >, rcvref< TO > > )
        return val;

    if ( val < std::numeric_limits< TO >::lowest() )
        return std::numeric_limits< TO >::lowest();
    if ( std::numeric_limits< TO >::max() < val )
        return std::numeric_limits< TO >::max();
    if ( -std::numeric_limits< TO >::min() < val &&
        val < std::numeric_limits< TO >::min() )
        return std::signbit( val )
        ? -std::numeric_limits< TO >::min()
        : std::numeric_limits< TO >::min();

    return static_cast< TO >( val );
}


template < typename T >
constexpr T squared( const T& x )
{
    return x * x;
}


constexpr space_t radians( const space_t& degrees )
{
    return SP_PI_180 * degrees;
}


constexpr bool abs_less(
    const space_t& smaller,
    const space_t& bigger
)
{
    return std::isless( std::fabs( smaller ), std::fabs( bigger ) );
}


constexpr bool almost_zero( const space_t& x )
{
    return ( x == 0 ) ? true : std::islessequal( std::fabs( x ), SP_EPSILON );
}


constexpr space_t clamp_epsilon_0( const space_t& x )
{
    return almost_zero( x ) ? 0 : x;
}


constexpr void running_compensation(
    const space_t& number,
    space_t& sum,
    space_t& err,
    space_t& temp
)
{
    // See https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Precision
    temp = sum + number;
    err += abs_less( number, sum )
        ? ( sum - temp ) + number
        : ( number - temp ) + sum;
    sum = temp;
}


constexpr void compensated_sum_v(
    space_t& sum, space_t& err, space_t& temp,
    const space_t& tail
)
{
    running_compensation( tail, sum, err, temp );
}


template < typename... Targs >
constexpr void compensated_sum_v(
    space_t& sum, space_t& err, space_t& temp,
    const space_t& head, Targs&&... rest
)
{
    compensated_sum_v( sum, err, temp, std::forward< Targs >( rest )... );
    running_compensation( head, sum, err, temp );
}


template < typename... Targs >
constexpr space_t compensated_sum( Targs&&... numbers )
{
    space_t sum = 0.0, err = 0.0, temp = 0.0;
    compensated_sum_v( sum, err, temp, std::forward< Targs >( numbers )... );
    return clamp_epsilon_0( sum + err );
}


constexpr bool almost_equal(
    const space_t& a,
    const space_t& b
)
{
    // Trivial case
    if ( a == b )
        return true;

    // Check if the numbers are really close -- needed
    // when comparing numbers near zero.
    const space_t diff = std::fabs( a - b );
    if ( std::islessequal( diff, SP_EPSILON ) )
        return true;

    // Sign comparison
    if ( std::signbit( a ) != std::signbit( b ) )
        return false;

    // Relative epsilon difference comparison
    return std::islessequal(
        diff * SP_RELATIVE_EPSILON, std::fabs( a ) + std::fabs( b )
    );
}


constexpr bool interval_test(
    const space_t& lower_bound,
    const space_t& value,
    const space_t& upper_bound
)
{
    return ( almost_equal( lower_bound, value ) || almost_equal( value, upper_bound ) )
        ? true
        : std::isless( lower_bound, value ) && std::isless( value, upper_bound );
}


constexpr bool leq_test(
    const space_t& smaller,
    const space_t& bigger
)
{
    return almost_equal( smaller, bigger ) ? true : std::isless( smaller, bigger );
}


constexpr space_t determinant3x3(
    const space_t& a, const space_t& b, const space_t& c,
    const space_t& d, const space_t& e, const space_t& f,
    const space_t& g, const space_t& h, const space_t& i
)
{
    // Using Laplace expansion on first row
    return compensated_sum(
        a * e * i,
        -( a * f * h ),
        -( b * d * i ),
        b * f * g,
        c * d * h,
        -( c * e * g )
    );
}
}


#endif
