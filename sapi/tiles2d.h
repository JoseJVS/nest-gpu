#ifndef TILES2D_H
#define TILES2D_H

#include "tile.h"
#include "coordinate_geometry.h"

// Third party
#include "random_generators.h"


namespace sapi
{
class Tile2D : public Tile< Coord2D >
{
public:
    Tile2D() = delete;
    Tile2D( const Tile& ) = delete;
    Tile2D( Tile2D&& ) = default;

    Tile2D(
        const Coord2D& origin,
        const space_t& radius,
        const vertidx_t& num_vertices,
        const space_t& angular_rotation,
        const space_t& angular_offset
    )
        : Tile< Coord2D >(
            0, // index
            origin,
            radius
        )
    {
        _initialize_vertices(
            origin,
            radius,
            num_vertices,
            angular_rotation,
            angular_offset
        );
    }

    Tile2D(
        tileidx_t&& index,
        CircumscribedRadius< Coord2D >&& c_radius,
        std::vector< Coord2D >&& vertices
    )
        : Tile< Coord2D >(
            std::move( index ),
            std::move( c_radius ),
            std::move( vertices )
        )
    {
    }

    virtual bool coord_in_tile(
        const Coord2D& coord
    ) const override;

    Coord2D project_point_to_surface(
        const Coord2D& coord
    ) const override;

protected:
    void _initialize_vertices(
        const Coord2D& origin,
        const space_t& radius,
        const vertidx_t& num_vertices,
        const space_t& angular_rotation,
        const space_t& angular_offset
    );

    virtual void _initialize_projection_vectors();

    bool triangular_projections_ = false;
    space_t projections_det_;
    std::vector< Coord2D > projection_vectors_;
};


class Square final : public Tile2D
{
public:
    Square() = delete;
    Square( const Square& ) = delete;
    Square( Square&& ) = default;

    Square(
        const Coord2D& origin,
        const space_t& radius,
        const space_t& angular_offset = 45
    )
        : Tile2D(
            origin,
            radius,
            4, // num vertices
            90, // base rotation
            angular_offset
        )
    {
        _initialize_projection_vectors();
        triangular_projections_ = false;
    }

    Square(
        tileidx_t&& index,
        CircumscribedRadius< Coord2D >&& c_radius,
        std::vector< Coord2D >&& vertices
    )
        : Tile2D(
            std::move( index ),
            std::move( c_radius ),
            std::move( vertices )
        )
    {
        _initialize_projection_vectors();
        triangular_projections_ = false;
    }

    void initialize_sub_tiles(
        const split_t& splits
    ) override;

    tileidx_t compute_sub_tile_count(
        const split_t& splits
    ) const override;

    void generate_coords_in_tile(
        std::vector< Coord2D >& coord_vec,
        nest::RngPtr const& rng
    ) const override;

    std::vector< split_t >
        get_possible_sub_tile_branches(
            const split_t& known_split_order
        ) const override;

    tileidx_t
        compute_leaf_index(
            const std::vector< split_t >& branch_sequence
        ) const override;

    std::string get_name() const override;
};


class Triangle final : public Tile2D
{
public:
    Triangle() = delete;
    Triangle( const Triangle& ) = delete;
    Triangle( Triangle&& ) = default;

    Triangle(
        const Coord2D& origin,
        const space_t& radius,
        const space_t& angular_offset = 45
    )
        : Tile2D(
            origin,
            radius,
            3, // num vertices
            90, // base rotation
            angular_offset
        )
    {
        _initialize_projection_vectors();
        triangular_projections_ = true;
    }

    Triangle(
        tileidx_t&& index,
        CircumscribedRadius< Coord2D >&& c_radius,
        std::vector< Coord2D >&& vertices
    )
        : Tile2D(
            std::move( index ),
            std::move( c_radius ),
            std::move( vertices )
        )
    {
        _initialize_projection_vectors();
        triangular_projections_ = true;
    }

    void initialize_sub_tiles(
        const split_t& splits
    ) override;

    tileidx_t compute_sub_tile_count(
        const split_t& splits
    ) const override;

    void generate_coords_in_tile(
        std::vector< Coord2D >& coord_vec,
        nest::RngPtr const& rng
    ) const override;

    std::vector< split_t >
        get_possible_sub_tile_branches(
            const split_t& known_split_order
        ) const override;

    tileidx_t
        compute_leaf_index(
            const std::vector< split_t >& branch_sequence
        ) const override;

    std::string get_name() const override;
};


class Hexagon final : public Tile2D
{
public:
    Hexagon() = delete;
    Hexagon( const Hexagon& ) = delete;
    Hexagon( Hexagon&& ) = default;

    Hexagon(
        const Coord2D& origin,
        const space_t& radius,
        const space_t& angular_offset = 30
    )
        : Tile2D(
            origin,
            radius,
            6, // num vertices
            60, // base rotation
            angular_offset
        )
    {
        _initialize_projection_vectors();
        triangular_projections_ = false;
    }

    void initialize_sub_tiles(
        const split_t& splits
    ) override;

    bool coord_in_tile(
        const Coord2D& coord
    ) const override;

    tileidx_t compute_sub_tile_count(
        const split_t& splits
    ) const override;

    void generate_coords_in_tile(
        std::vector< Coord2D >& coord_vec,
        nest::RngPtr const& rng
    ) const override;

    std::vector< split_t >
        get_possible_sub_tile_branches(
            const split_t& known_split_order
        ) const override;

    tileidx_t
        compute_leaf_index(
            const std::vector< split_t >& branch_sequence
        ) const override;

    std::string get_name() const override;

protected:
    void _initialize_projection_vectors() override;
};


constexpr tileidx_t
compute_sub_tile_split_index(
    const tileidx_t& base_index,
    const split_t& tile_position,
    const split_t& split_power,
    const split_t& remaining_splits
)
{
    return base_index + 1
        + static_cast< tileidx_t >( tile_position )
        * std::pow( split_power, remaining_splits );
}


inline tileidx_t
compute_tile_index_from_branching_sequence(
    const split_t& split_power,
    const std::vector< split_t >& branch_sequence
)
{
    if ( branch_sequence.empty() )
        return 0;
    else
    {
        assert( branch_sequence.size() < std::numeric_limits< split_t >::max() );
        split_t rem_splits = static_cast< split_t >( branch_sequence.size() );
        tileidx_t index = 0;
        for ( const auto& branch_pos : branch_sequence )
            index = compute_sub_tile_split_index( index, branch_pos, split_power, --rem_splits );
        assert( 0 < index );
        return index;
    }
}


inline void
Tile2D::_initialize_vertices(
    const Coord2D& origin,
    const space_t& radius,
    const vertidx_t& num_vertices,
    const space_t& angular_rotation,
    const space_t& angular_offset
)
{
    assert( 0 < num_vertices && !almost_zero( angular_rotation ) );
    vertices_.reserve( num_vertices );
    std::function< space_t( const vertidx_t& ) >
        total_rotation = [ & ]( const vertidx_t& angle_ix )
        { return angular_rotation * angle_ix; };
    if ( !almost_zero( angular_offset ) )
        total_rotation = [ & ]( const vertidx_t& angle_ix )
        { return compensated_sum( angular_offset, angular_rotation * angle_ix ); };
    for ( vertidx_t angle_ix = 0; angle_ix < num_vertices; ++angle_ix )
        vertices_.emplace_back( origin + ( create_angular_offset( total_rotation( angle_ix ) ) * radius ) );
}


inline void
Tile2D::_initialize_projection_vectors()
{
    assert( 2 < vertices_.size() && projection_vectors_.empty() );
    projection_vectors_.reserve( 2 );
    projection_vectors_.emplace_back( vertices_[ 0 ] - vertices_[ 1 ] );
    projection_vectors_.emplace_back( vertices_[ 2 ] - vertices_[ 1 ] );

    projections_det_ = coord_sum(
        vector_cross( projection_vectors_[ 0 ], projection_vectors_[ 1 ] )
    );
    assert( !almost_zero( projections_det_ ) );
}


inline bool
Tile2D::coord_in_tile(
    const Coord2D& coord
) const
{
    if ( !c_radius_.coord_in_radius( coord ).has_value() )
        return false;

    // Only works for squares or triangles
    return algebraic_projection_comparison(
        coord - vertices_[ 1 ],
        projection_vectors_[ 0 ],
        projection_vectors_[ 1 ],
        projections_det_,
        triangular_projections_
    );
}


inline Coord2D
Tile2D::project_point_to_surface(
    const Coord2D& coord
) const
{
    vertidx_t v_index = 0;
    vertidx_t first = 0;
    space_t d_first = std::numeric_limits< space_t >::max();
    vertidx_t second = 0;
    space_t d_second = std::numeric_limits< space_t >::max();
    for ( const auto& vertex : vertices_ )
    {
        const auto d2 = distance2( coord, vertex );
        if ( leq_test( d2, d_first ) )
        {
            second = first;
            d_second = d_first;
            first = v_index;
            d_first = d2;
        }
        else if ( leq_test( d2, d_second ) )
        {
            second = v_index;
            d_second = d2;
        }
        ++v_index;
    }

    if ( almost_zero( d_first ) )
        return vertices_[ first ];

    return projection_coord(
        coord,
        vertices_[ first ],
        vertices_[ second ],
        true // Clamped projection
    );
}


inline tileidx_t
Square::compute_sub_tile_count(
    const split_t& splits
) const
{
    return std::pow( 4, splits );
}


inline std::vector< split_t >
Square::get_possible_sub_tile_branches(
    const split_t& known_split_order
) const
{
    if ( known_split_order == 0 )
        return {};
    else
    {
        return std::vector< split_t >( known_split_order, 4 );
    }
}


inline tileidx_t
Square::compute_leaf_index(
    const std::vector< split_t >& branch_sequence
) const
{
    return compute_tile_index_from_branching_sequence(
        5, branch_sequence
    );
}


inline std::string
Square::get_name() const
{
    return "Square";
}


inline void
Square::initialize_sub_tiles( const split_t& splits )
{
    if ( splits < 1 || !sub_tiles_.empty() )
        return;

    sub_tiles_.reserve( 4 );
    const split_t rem_splits = splits - 1;
    const space_t radius2 = c_radius_.radius2_ / 4;

    Coord2D midpoints[ 4 ];
    for ( vertidx_t vix = 0; vix < 4; ++vix )
        midpoints[ vix ] = midpoint( vertices_[ vix ], vertices_[ ( vix + 1 ) % 4 ] );

    Coord2D new_origin;
    vertidx_t opposite_midpoint;
    for ( vertidx_t partition = 0; partition < 4; ++partition )
    {
        opposite_midpoint = ( partition + 3 ) % 4;
        new_origin = midpoint( c_radius_.origin_, vertices_[ partition ] );

        sub_tiles_.emplace_back( std::make_unique< Square >( Square(
            compute_sub_tile_split_index( index_, partition, 5, rem_splits ),
            CircumscribedRadius< Coord2D >(
                new_origin,
                radius2
            ),
            { vertices_[ partition ],
            midpoints[ partition ],
            c_radius_.origin_,
            midpoints[ opposite_midpoint ] }
        ) ) );
    }

#pragma omp taskloop num_tasks( 4 ) mergeable final( rem_splits < 4 )\
default( none ) shared( sub_tiles_ ) firstprivate( rem_splits )
    for ( vertidx_t partition = 0; partition < 4; ++partition )
        sub_tiles_[ partition ]->initialize_sub_tiles( rem_splits );
}


inline void
Square::generate_coords_in_tile(
    std::vector< Coord2D >& coord_vec,
    nest::RngPtr const& rng
) const
{
    space_t m0, m1;
    for ( auto coord_vec_it = coord_vec.begin();
        coord_vec_it != coord_vec.end(); ++coord_vec_it )
    {
        m0 = static_cast< space_t >( rng->drand() );
        m1 = static_cast< space_t >( rng->drand() );
        *coord_vec_it = Coord2D(
            compensated_sum(
                vertices_[ 1 ].x_,
                projection_vectors_[ 0 ].x_ * m0,
                projection_vectors_[ 1 ].x_ * m1
            ),
            compensated_sum(
                vertices_[ 1 ].y_,
                projection_vectors_[ 0 ].y_ * m0,
                projection_vectors_[ 1 ].y_ * m1
            )
        );
    }
}


inline tileidx_t
Triangle::compute_sub_tile_count(
    const split_t& splits
) const
{
    return std::pow( 2, splits );
}


inline std::vector< split_t >
Triangle::get_possible_sub_tile_branches(
    const split_t& known_split_order
) const
{
    if ( known_split_order == 0 )
        return {};
    else
    {
        return std::vector< split_t >( known_split_order, 2 );
    }
}


inline tileidx_t
Triangle::compute_leaf_index(
    const std::vector< split_t >& branch_sequence
) const
{
    return compute_tile_index_from_branching_sequence(
        3, branch_sequence
    );
}


inline std::string
Triangle::get_name() const
{
    return "Triangle";
}


inline void
Triangle::initialize_sub_tiles(
    const split_t& splits
)
{
    if ( splits < 1 || !sub_tiles_.empty() )
        return;

    sub_tiles_.reserve( 2 );
    const split_t rem_splits = splits - 1;

    // Get AB, the largest edge of triangle ABC
    Coord2D coordA, coordB, coordC;
    {
        std::array< Coord2D, 3 > sorted_coords;
        std::array< Coord2D, 3 > sorted_vectors;
        std::array< space_t, 3 > sorted_distances2;
        sort_vertices(
            { vertices_[ 0 ], vertices_[ 1 ], vertices_[ 2 ] },
            sorted_coords,
            sorted_vectors,
            sorted_distances2
        );
        coordA = std::move( sorted_coords[ 0 ] );
        coordB = std::move( sorted_coords[ 1 ] );
        coordC = std::move( sorted_coords[ 2 ] );
    }

    // Get D midpoint of AB,
    // this midpoint is now a vertex of each sub-triangle DCA and DCB,
    // this guarantees each sub-triangle has same area size.
    const Coord2D coordD( midpoint( coordA, coordB ) );

    sub_tiles_.emplace_back( std::make_unique< Triangle >( Triangle(
        index_ + 1,
        compute_c_radius( coordC, coordD, coordA ),
        { coordC,
        coordD,
        coordA }
    ) ) );
    sub_tiles_.emplace_back( std::make_unique< Triangle >( Triangle(
        compute_sub_tile_split_index( index_, 1, 3, rem_splits ),
        compute_c_radius( coordC, coordD, coordB ),
        { coordC,
        coordD,
        coordB }
    ) ) );

#pragma omp taskloop num_tasks( 2 ) mergeable final( rem_splits < 8 )\
default( none ) shared( sub_tiles_ ) firstprivate( rem_splits )
    for ( vertidx_t partition = 0; partition < 2; ++partition )
        sub_tiles_[ partition ]->initialize_sub_tiles( rem_splits );
}


inline void
Triangle::generate_coords_in_tile(
    std::vector< Coord2D >& coord_vec,
    nest::RngPtr const& rng
) const
{
    space_t m0, m1, m2;
    for ( auto coord_vec_it = coord_vec.begin();
        coord_vec_it != coord_vec.end(); ++coord_vec_it )
    {
        m0 = static_cast< space_t >( rng->drand() );
        m1 = static_cast< space_t >( rng->drand() );
        if ( std::isless( 1., m0 + m1 ) )
        {
            m0 = std::fmax( std::fmin( 1., 1. - m0 ), 0. );
            m1 = std::fmax( std::fmin( 1., 1. - m1 ), 0. );
        }
        m2 = std::fmax( std::fmin( 1., 1. - m0 - m1 ), 0. );
        *coord_vec_it = Coord2D(
            compensated_sum(
                vertices_[ 0 ].x_ * m0,
                vertices_[ 1 ].x_ * m1,
                vertices_[ 2 ].x_ * m2
            ),
            compensated_sum(
                vertices_[ 0 ].y_ * m0,
                vertices_[ 1 ].y_ * m1,
                vertices_[ 2 ].y_ * m2
            )
        );
    }
}


inline tileidx_t
Hexagon::compute_sub_tile_count(
    const split_t& splits
) const
{
    return splits > 0 ? 6 * std::pow( 2, splits - 1 ) : 1;
}


inline std::vector< split_t >
Hexagon::get_possible_sub_tile_branches(
    const split_t& known_split_order
) const
{
    if ( known_split_order == 0 )
        return {};
    else
    {
        std::vector< split_t > splits( known_split_order, 2 );
        splits[ 0 ] = 6;
        return splits;
    }
}


inline tileidx_t
Hexagon::compute_leaf_index(
    const std::vector< split_t >& branch_sequence
) const
{
    return compute_tile_index_from_branching_sequence(
        3, branch_sequence
    );
}


inline std::string
Hexagon::get_name() const
{
    return "Hexagon";
}


inline void
Hexagon::_initialize_projection_vectors()
{
    assert( !vertices_.empty() && projection_vectors_.empty() );

    // Hexagon is divided into rectangles composed of opposing edges
    // used for parallelogram projection comparison
    projection_vectors_.reserve( 6 );
    for ( vertidx_t vix = 0; vix < 3; ++vix )
    {
        projection_vectors_.emplace_back(
            vertices_[ vix + 1 ] - vertices_[ vix ]
        );
        projection_vectors_.emplace_back(
            vertices_[ ( vix + 4 ) % 6 ] - vertices_[ vix ]
        );
    }

    projections_det_ = coord_sum(
        vector_cross( projection_vectors_[ 0 ], projection_vectors_[ 1 ] )
    );
    assert( !almost_zero( projections_det_ ) );
}


inline void
Hexagon::initialize_sub_tiles(
    const split_t& splits
)
{
    if ( splits < 1 || !sub_tiles_.empty() )
        return;

    sub_tiles_.reserve( 6 );
    const split_t rem_splits = splits - 1;
    const space_t radius2 = c_radius_.radius2_ / 3;

    Coord2D new_origin;
    vertidx_t next_partition;
    for ( vertidx_t partition = 0; partition < 6; ++partition )
    {
        next_partition = ( partition + 1 ) % 6;

        new_origin = ( c_radius_.origin_ / 3 )
            + ( ( vertices_[ partition ] + vertices_[ next_partition ] ) / 3 );

        sub_tiles_.emplace_back( std::make_unique< Triangle >( Triangle(
            compute_sub_tile_split_index( index_, partition, 3, rem_splits ),
            CircumscribedRadius< Coord2D >(
                new_origin,
                radius2
            ),
            { c_radius_.origin_,
            vertices_[ partition ],
            vertices_[ next_partition ] }
        ) ) );
    }

#pragma omp taskloop num_tasks( 3 ) mergeable final( rem_splits < 9 )\
default( none ) shared( sub_tiles_ ) firstprivate( rem_splits )
    for ( vertidx_t partition = 0; partition < 6; ++partition )
        sub_tiles_[ partition ]->initialize_sub_tiles( rem_splits );
}


inline bool
Hexagon::coord_in_tile(
    const Coord2D& coord
) const
{
    if ( !c_radius_.coord_in_radius( coord ).has_value() )
        return false;

    // Parallelogram based comparison using rotating rectangles
    bool res = false;
    vertidx_t rel_vix;
    for ( vertidx_t vix = 0; vix < 3; ++vix )
    {
        rel_vix = 2 * vix;
        res = algebraic_projection_comparison(
            coord - vertices_[ vix ],
            projection_vectors_[ rel_vix ],
            projection_vectors_[ rel_vix + 1 ],
            projections_det_,
            false // Triangular comparison
        );
        if ( res ) break;
    }
    return res;
}


inline void
Hexagon::generate_coords_in_tile(
    std::vector< Coord2D >& coord_vec,
    nest::RngPtr const& rng
) const
{
    nest::uniform_int_distribution hex_idx_dist;
    hex_idx_dist.param(
        nest::uniform_int_distribution::param_type(
            0, 5
        )
    );
    nest::uniform_int_distribution::result_type hex_idx;
    space_t m0, m1, m2;
    for ( auto coord_vec_it = coord_vec.begin();
        coord_vec_it != coord_vec.end(); ++coord_vec_it )
    {
        m0 = static_cast< space_t >( rng->drand() );
        m1 = static_cast< space_t >( rng->drand() );
        if ( std::isless( 1., m0 + m1 ) )
        {
            m0 = std::fmax( std::fmin( 1., 1. - m0 ), 0. );
            m1 = std::fmax( std::fmin( 1., 1. - m1 ), 0. );
        }
        m2 = std::fmax( std::fmin( 1., 1. - m0 - m1 ), 0. );
        hex_idx = hex_idx_dist( rng );
        *coord_vec_it = Coord2D(
            compensated_sum(
                c_radius_.origin_.x_ * m0,
                vertices_[ hex_idx ].x_ * m1,
                vertices_[ ( hex_idx + 1 ) % 6 ].x_ * m2
            ),
            compensated_sum(
                c_radius_.origin_.y_ * m0,
                vertices_[ hex_idx ].y_ * m1,
                vertices_[ ( hex_idx + 1 ) % 6 ].y_ * m2
            )
        );
    }
}
}


#endif
