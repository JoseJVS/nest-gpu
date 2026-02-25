#ifndef COORDINATES_H
#define COORDINATES_H

#include <string>
#include <optional>
#include <cstring>

#include "numerics.h"


namespace sapi
{
struct Coord2D
{
    constexpr static const dim_t D = 2;
    space_t x_ = 0.;
    space_t y_ = 0.;

    Coord2D() = default;
    Coord2D( const Coord2D& ) = default;
    Coord2D( Coord2D&& ) = default;
    ~Coord2D() = default;

    Coord2D( const space_t& x, const space_t& y )
        : x_( x ), y_( y )
    {
    }

    Coord2D( space_t&& x, space_t&& y )
        : x_( std::move( x ) ), y_( std::move( y ) )
    {
    }

    Coord2D& operator=( const Coord2D& coord );

    Coord2D& operator=( Coord2D&& coord );

    Coord2D operator+( const Coord2D& coord ) const;

    Coord2D operator+( const space_t& scalar ) const;

    Coord2D operator-( const Coord2D& coord ) const;

    Coord2D operator-( const space_t& scalar ) const;

    Coord2D operator*( const space_t& scalar ) const;

    Coord2D operator*( const Coord2D& coord ) const;

    Coord2D operator/( const space_t& scalar ) const;

    Coord2D operator/( const Coord2D& coord ) const;

    bool operator==( const Coord2D& coord ) const;

    bool is_null() const;

    Coord2D negative() const;

    std::string to_string() const;

    Coord2D& sanitize();

    template < typename IT >
    void copy_to_vec( IT&& writing_pos ) const;

    template < typename IT >
    static Coord2D copy_from_vec( IT&& reading_pos );

    template < typename IT >
    void bit_copy_to_vec( IT&& writing_pos ) const;

    template < typename IT >
    static Coord2D bit_copy_from_vec( IT&& reading_pos );
};


inline Coord2D& Coord2D::operator=( const Coord2D& coord )
{
    x_ = coord.x_;
    y_ = coord.y_;
    return *this;
}


inline Coord2D& Coord2D::operator=( Coord2D&& coord )
{
    x_ = std::move( coord.x_ );
    y_ = std::move( coord.y_ );
    return *this;
}


inline Coord2D Coord2D::operator+( const Coord2D& coord ) const
{
    return Coord2D(
        compensated_sum( x_, coord.x_ ),
        compensated_sum( y_, coord.y_ )
    );
}


inline Coord2D Coord2D::operator+( const space_t& scalar ) const
{
    return Coord2D(
        compensated_sum( x_, scalar ),
        compensated_sum( y_, scalar )
    );
}


inline Coord2D Coord2D::operator-( const Coord2D& coord ) const
{
    return Coord2D(
        compensated_sum( x_, -coord.x_ ),
        compensated_sum( y_, -coord.y_ )
    );
}


inline Coord2D Coord2D::operator-( const space_t& scalar ) const
{
    return Coord2D(
        compensated_sum( x_, -scalar ),
        compensated_sum( y_, -scalar )
    );
}


inline Coord2D Coord2D::operator*( const space_t& scalar ) const
{
    return Coord2D(
        x_ * scalar,
        y_ * scalar
    ).sanitize();
}


inline Coord2D Coord2D::operator*( const Coord2D& coord ) const
{
    return Coord2D(
        x_ * coord.x_,
        y_ * coord.y_
    ).sanitize();
}


inline Coord2D Coord2D::operator/( const space_t& scalar ) const
{
    return Coord2D(
        x_ / scalar,
        y_ / scalar
    ).sanitize();
}


inline Coord2D Coord2D::operator/( const Coord2D& coord ) const
{
    return Coord2D(
        x_ / coord.x_,
        y_ / coord.y_
    ).sanitize();
}


inline bool Coord2D::operator==( const Coord2D& coord ) const
{
    return almost_equal( x_, coord.x_ ) && almost_equal( y_, coord.y_ );
}


inline bool Coord2D::is_null() const
{
    return almost_zero( x_ ) && almost_zero( y_ );
}


inline Coord2D Coord2D::negative() const
{
    return Coord2D( -x_, -y_ );
}


inline std::string Coord2D::to_string() const
{
    return "{ x: " + std::to_string( x_ )
        + " , y: " + std::to_string( y_ ) + " }";
}


inline Coord2D& Coord2D::sanitize()
{
    x_ = clamp_epsilon_0( x_ );
    y_ = clamp_epsilon_0( y_ );
    return *this;
}


template < typename IT >
inline void Coord2D::copy_to_vec( IT&& writing_pos ) const
{
    *writing_pos++ = x_;
    *writing_pos++ = y_;
}


template < typename IT >
inline Coord2D Coord2D::copy_from_vec( IT&& reading_pos )
{
    return Coord2D(
        space_t( *reading_pos++ ),
        space_t( *reading_pos++ )
    );
}


template < typename IT >
inline void Coord2D::bit_copy_to_vec( IT&& writing_pos ) const
{
    std::memcpy( &( *writing_pos++ ), &x_, sizeof( space_t ) );
    std::memcpy( &( *writing_pos++ ), &y_, sizeof( space_t ) );
}


template < typename IT >
inline Coord2D Coord2D::bit_copy_from_vec( IT&& reading_pos )
{
    Coord2D coord;
    std::memcpy( &coord.x_, &( *reading_pos++ ), sizeof( space_t ) );
    std::memcpy( &coord.y_, &( *reading_pos++ ), sizeof( space_t ) );
    return coord;
}


struct Coord3D
{
    constexpr static const dim_t D = 3;
    space_t x_ = 0.;
    space_t y_ = 0.;
    space_t z_ = 0.;

    Coord3D() = default;
    Coord3D( const Coord3D& coord ) = default;
    Coord3D( Coord3D&& coord ) = default;
    ~Coord3D() = default;

    Coord3D( const space_t& x, const space_t& y, const space_t& z )
        : x_( x ), y_( y ), z_( z )
    {
    }

    Coord3D( space_t&& x, space_t&& y, space_t&& z )
        : x_( std::move( x ) ), y_( std::move( y ) ), z_( std::move( z ) )
    {
    }

    Coord3D& operator=( const Coord3D& coord );

    Coord3D& operator=( Coord3D&& coord );

    Coord3D operator+( const space_t& scalar ) const;

    Coord3D operator+( const Coord3D& coord ) const;

    Coord3D operator-( const space_t& scalar ) const;

    Coord3D operator-( const Coord3D& coord ) const;

    Coord3D operator*( const space_t& scalar ) const;

    Coord3D operator*( const Coord3D& coord ) const;

    Coord3D operator/( const space_t& scalar ) const;

    Coord3D operator/( const Coord3D& coord ) const;

    bool operator==( const Coord3D& coord ) const;

    bool is_null() const;

    Coord3D negative() const;

    std::string to_string() const;

    Coord3D& sanitize();

    template < typename IT >
    void copy_to_vec( IT&& writing_pos ) const;

    template < typename IT >
    static Coord3D copy_from_vec( IT&& reading_pos );

    template < typename IT >
    void bit_copy_to_vec( IT&& writing_pos ) const;

    template < typename IT >
    static Coord3D bit_copy_from_vec( IT&& reading_pos );
};


inline Coord3D& Coord3D::operator=( const Coord3D& coord )
{
    x_ = coord.x_;
    y_ = coord.y_;
    z_ = coord.z_;
    return *this;
}


inline Coord3D& Coord3D::operator=( Coord3D&& coord )
{
    x_ = std::move( coord.x_ );
    y_ = std::move( coord.y_ );
    z_ = std::move( coord.z_ );
    return *this;
}


inline Coord3D Coord3D::operator+( const Coord3D& coord ) const
{
    return Coord3D(
        compensated_sum( x_, coord.x_ ),
        compensated_sum( y_, coord.y_ ),
        compensated_sum( z_, coord.z_ )
    );
}


inline Coord3D Coord3D::operator+( const space_t& scalar ) const
{
    return Coord3D(
        compensated_sum( x_, scalar ),
        compensated_sum( y_, scalar ),
        compensated_sum( z_, scalar )
    );
}


inline Coord3D Coord3D::operator-( const Coord3D& coord ) const
{
    return Coord3D(
        compensated_sum( x_, -coord.x_ ),
        compensated_sum( y_, -coord.y_ ),
        compensated_sum( z_, -coord.z_ )
    );
}


inline Coord3D Coord3D::operator-( const space_t& scalar ) const
{
    return Coord3D(
        compensated_sum( x_, -scalar ),
        compensated_sum( y_, -scalar ),
        compensated_sum( z_, -scalar )
    );
}


inline Coord3D Coord3D::operator*( const space_t& scalar ) const
{
    return Coord3D(
        x_ * scalar,
        y_ * scalar,
        z_ * scalar
    ).sanitize();
}


inline Coord3D Coord3D::operator*( const Coord3D& coord ) const
{
    return Coord3D(
        x_ * coord.x_,
        y_ * coord.y_,
        z_ * coord.z_
    ).sanitize();
}


inline Coord3D Coord3D::operator/( const space_t& scalar ) const
{
    return Coord3D(
        x_ / scalar,
        y_ / scalar,
        z_ / scalar
    ).sanitize();
}


inline Coord3D Coord3D::operator/( const Coord3D& coord ) const
{
    return Coord3D(
        x_ / coord.x_,
        y_ / coord.y_,
        z_ / coord.z_
    ).sanitize();
}


inline bool Coord3D::operator==( const Coord3D& coord ) const
{
    return almost_equal( x_, coord.x_ ) && almost_equal( y_, coord.y_ ) && almost_equal( z_, coord.z_ );
}


inline bool Coord3D::is_null() const
{
    return almost_zero( x_ ) && almost_zero( y_ ) && almost_zero( z_ );
}


inline Coord3D Coord3D::negative() const
{
    return Coord3D( -x_, -y_, -z_ );
}


inline std::string Coord3D::to_string() const
{
    return "{ x: " + std::to_string( x_ )
        + " , y: " + std::to_string( y_ )
        + " , z: " + std::to_string( z_ ) + " }";
}


inline Coord3D& Coord3D::sanitize()
{
    x_ = clamp_epsilon_0( x_ );
    y_ = clamp_epsilon_0( y_ );
    z_ = clamp_epsilon_0( z_ );
    return *this;
}


template < typename IT >
inline void Coord3D::copy_to_vec( IT&& writing_pos ) const
{
    *writing_pos++ = x_;
    *writing_pos++ = y_;
    *writing_pos++ = z_;
}


template < typename IT >
inline Coord3D Coord3D::copy_from_vec( IT&& reading_pos )
{
    return Coord3D(
        space_t( *reading_pos++ ),
        space_t( *reading_pos++ ),
        space_t( *reading_pos++ )
    );
}


template < typename IT >
inline void Coord3D::bit_copy_to_vec( IT&& writing_pos ) const
{
    std::memcpy( &( *writing_pos++ ), &x_, sizeof( space_t ) );
    std::memcpy( &( *writing_pos++ ), &y_, sizeof( space_t ) );
    std::memcpy( &( *writing_pos++ ), &z_, sizeof( space_t ) );
}


template < typename IT >
inline Coord3D Coord3D::bit_copy_from_vec( IT&& reading_pos )
{
    Coord3D coord;
    std::memcpy( &coord.x_, &( *reading_pos++ ), sizeof( space_t ) );
    std::memcpy( &coord.y_, &( *reading_pos++ ), sizeof( space_t ) );
    std::memcpy( &coord.z_, &( *reading_pos++ ), sizeof( space_t ) );
    return coord;
}


template < typename CoordT >
struct Displacement
{
    CoordT displacement_;
    space_t distance2_ = 0.;

    Displacement() = default;
    Displacement( const Displacement& dc ) = default;
    Displacement( Displacement&& dc ) = default;
    ~Displacement() = default;

    Displacement( CoordT&& );

    Displacement& operator=( const Displacement& );

    Displacement& operator=( Displacement&& );

    space_t get_distance() const;

    bool operator==( const Displacement& ) const;
};


// Forward definition to link with coordinate_geometry.h
template < typename CoordT >
space_t vector_norm2( const CoordT& );


template < typename CoordT >
Displacement< CoordT >::Displacement( CoordT&& displacement )
    : displacement_( std::move( displacement ) )
    , distance2_( vector_norm2( displacement_ ) )
{
}


template < typename CoordT >
inline Displacement< CoordT >& Displacement< CoordT >::operator=(
    const Displacement< CoordT >& dc
    )
{
    displacement_ = dc.displacement_;
    distance2_ = dc.distance2_;
    return *this;
}


template < typename CoordT >
inline Displacement< CoordT >& Displacement< CoordT >::operator=(
    Displacement&& dc
    )
{
    displacement_ = std::move( dc.displacement_ );
    distance2_ = std::move( dc.distance2_ );
    return *this;
}


template < typename CoordT >
inline space_t Displacement< CoordT >::get_distance() const
{
    return std::sqrt( distance2_ );
}


template < typename CoordT >
inline bool Displacement< CoordT >::operator==( const Displacement< CoordT >& dc ) const
{
    return displacement_ == dc.displacement_;
}


template < typename CoordT >
struct CircumscribedRadius
{
    CoordT origin_;
    space_t radius2_ = 0.;

    CircumscribedRadius() = default;
    CircumscribedRadius(
        const CircumscribedRadius&
    ) = default;
    CircumscribedRadius(
        CircumscribedRadius&&
    ) = default;
    ~CircumscribedRadius() = default;

    CircumscribedRadius(
        const CoordT& origin,
        const space_t& radius2
    )
        : origin_( origin )
        , radius2_( radius2 )
    {
    }

    CircumscribedRadius(
        CoordT&& origin,
        space_t&& radius2
    )
        : origin_( std::move( origin ) )
        , radius2_( std::move( radius2 ) )
    {
    }

    CircumscribedRadius& operator=(
        const CircumscribedRadius&
        );

    CircumscribedRadius& operator=(
        CircumscribedRadius&&
        );

    bool disp_in_radius(
        const Displacement< CoordT >& disp,
        const space_t& radius_offset = 0
    ) const;

    std::optional< Displacement< CoordT > >
        coord_in_radius(
            const CoordT& coord,
            const space_t& radius_offset = 0
        ) const;

    std::optional< Displacement< CoordT > >
        overlapping_radi(
            const CircumscribedRadius& radius
        ) const;

    std::string to_string() const;

    bool operator==(
        const CircumscribedRadius&
        ) const;
};


template < typename CoordT >
inline CircumscribedRadius< CoordT >&
CircumscribedRadius< CoordT >::operator=(
    const CircumscribedRadius< CoordT >& cr
    )
{
    origin_ = cr.origin_;
    radius2_ = cr.radius2_;
    return *this;
}


template < typename CoordT >
inline CircumscribedRadius< CoordT >&
CircumscribedRadius< CoordT >::operator=(
    CircumscribedRadius< CoordT >&& cr
    )
{
    origin_ = std::move( cr.origin_ );
    radius2_ = std::move( cr.radius2_ );
    return *this;
}


template < typename CoordT >
inline std::string CircumscribedRadius< CoordT >::to_string() const
{
    return "{ center: " + origin_.to_string() + " , radius: " + std::to_string( std::sqrt( radius2_ ) ) + " }";
}


template < typename CoordT >
inline bool CircumscribedRadius< CoordT >::disp_in_radius(
    const Displacement< CoordT >& disp,
    const space_t& radius_offset
) const
{
    return leq_test(
        disp.distance2_,
        almost_zero( radius_offset )
        ? radius2_
        : squared( compensated_sum( std::sqrt( radius2_ ), radius_offset ) )
    );
}


template < typename CoordT >
inline std::optional< Displacement< CoordT > >
CircumscribedRadius< CoordT >::coord_in_radius(
    const CoordT& coord,
    const space_t& radius_offset
) const
{
    Displacement< CoordT > disp( coord - origin_ );
    return disp_in_radius( disp, radius_offset )
        ? std::make_optional( std::move( disp ) )
        : std::optional< Displacement< CoordT > >();
}


template < typename CoordT >
inline std::optional< Displacement< CoordT > > CircumscribedRadius< CoordT >::overlapping_radi(
    const CircumscribedRadius& radius
) const
{
    return coord_in_radius(
        radius.origin_,
        std::sqrt( radius.radius2_ )
    );
}


template < typename CoordT >
inline bool CircumscribedRadius< CoordT >::operator==( const CircumscribedRadius& cr ) const
{
    return origin_ == cr.origin_ && almost_equal( radius2_, cr.radius2_ );
}
}


#endif
