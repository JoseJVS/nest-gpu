/*
 *  node_distribution.h
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

#ifndef NODE_DISTRIBUTION
#define NODE_DISTRIBUTION

#include "numerics.h"
#include "random_manager.h"
#include "node_containers.h"
#include "grid_neighborhood.h"


namespace sapi
{
// Forward definition to link with vp_interface
vp_t get_thread_num();
vp_t get_max_omp_threads();


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
std::vector< nodeidx_t >
safe_convert_vector(
    std::vector< I >&& count_vector
)
{
    if constexpr ( std::is_same_v< rcvref< I >, nodeidx_t > )
        return std::forward< std::vector< nodeidx_t > >( count_vector );

    std::vector< nodeidx_t > output_copy( count_vector.size() );
    auto out_it = output_copy.begin();
    for ( const auto& count : count_vector )
    {
        if (
            0 <= count &&
            count < std::numeric_limits< nodeidx_t >::max()
            )
            *out_it++ = static_cast< nodeidx_t >( count );
        else
            throw std::runtime_error( "Could not distribute node counts over available bins" );
    }

    count_vector.clear();

    return output_copy;
}


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
inline I
balance_node_counts_by_bins(
    I& num_nodes,
    const tileidx_t& num_bins,
    const bool& balanced
)
{
    if ( balanced )
    {
        const auto balancing = std::div( num_nodes, static_cast< I >( num_bins ) );
        num_nodes = balancing.rem;
        return balancing.quot;
    }

    return 0;
}


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
inline std::pair< I, I >
batch_node_counts_by_thread_count(
    const I& num_nodes,
    const vp_t& num_threads
)
{
    // Batch tile index drawing with steps equal to the total number of threads
    I full_batches = 0, partial_batches = 0;

    // Only batch if running in multithreaded environment
    if ( 1 < num_threads )
    {
        const auto batch_steps = std::div( num_nodes, static_cast< I >( num_threads ) );
        full_batches = batch_steps.quot;
        partial_batches = batch_steps.rem;
    }

    if ( full_batches < static_cast< I >( BATCHING_THRESHOLD ) )
    {
        full_batches = 0;
        partial_batches = num_nodes;
    }

    assert( ( full_batches * num_threads + partial_batches ) == num_nodes );

    return { full_batches, partial_batches };
}


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
void left_sum_vectors(
    std::vector< I >& left,
    std::vector< I >& right
)
{
    auto r_it = right.begin();
    for ( auto l_it = left.begin();
        l_it != left.end();
        ++l_it )
        *l_it += *r_it++;
    right.clear();
}


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
void left_merge_buffers(
    std::vector< std::vector< I > >& buffer_vec,
    const vp_t& num_buffers // Buffer size is always in number of threads
)
{
    vp_t step = 1, step2 = 2;
    do
    {
#pragma omp taskloop grainsize( 1 ) default( none )\
    shared( buffer_vec )\
    firstprivate( num_buffers, step, step2 )
        for ( vp_t curr_idx = 0; curr_idx < num_buffers; curr_idx += step2 )
        {
            const auto next_idx = curr_idx + step;
            if ( next_idx < num_buffers )
                left_sum_vectors( buffer_vec[ curr_idx ], buffer_vec[ next_idx ] );
        }
        step = step2;
        step2 *= 2;
    } while ( step < num_buffers );
}


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
void batched_draw(
    std::vector< I >& buffer,
    I& batches,
    nest::RngPtr const& rng,
    nest::uniform_int_distribution& distribution,
    const tileidx_t& num_bins,
    const bool& balanced
)
{
    if ( balanced )
    {
        assert( batches < num_bins && 1 < num_bins );
        std::vector< I > indexes( num_bins );
        std::iota( indexes.begin(), indexes.end(), 0 );
        rng->shuffle( indexes );

        while ( 0 < batches )
        {
            ++buffer[ indexes[ batches ] ];
            --batches;
        }
    }
    else
    {
        while ( 0 < batches )
        {
            ++buffer[ distribution( rng ) ];
            --batches;
        }
    }
}


template < std::size_t N  >
tileidx_t
collapse_dimensions(
    const std::vector< tileidx_t >& grid_pos,
    const std::array< tileidx_t, N >& grid_dimensions
)
{
    assert( grid_pos.size() == N );
    auto gp_it = grid_pos.rbegin();
    tileidx_t index = *gp_it++;
    for ( auto gd_it = grid_dimensions.rbegin() + 1;
        gd_it != grid_dimensions.rend();
        ++gd_it )
        index = *gp_it++ + *gd_it * index;
    return index;
}


template < typename SplitT,
    typename std::enable_if_t< std::is_integral_v< rcvref< SplitT > >, bool > = true
>
tileidx_t
collapse_dimensions(
    const std::vector< SplitT >& branch_sequence,
    const std::vector< SplitT >& split_sequence
)
{
    assert( branch_sequence.size() == split_sequence.size() );
    auto bs_it = branch_sequence.begin();
    tileidx_t index = *bs_it++;
    for ( auto ss_it = split_sequence.begin() + 1;
        ss_it != split_sequence.end();
        ++ss_it )
        index = *bs_it++ + *ss_it * index;
    return index;
}


template <
    typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true,
    typename ForwardDistIT,
    typename DimensionBoundsT
>
void batched_draw(
    std::vector< I >& buffer,
    I& batches,
    nest::RngPtr const& rng,
    const ForwardDistIT& distributions_first,
    const DimensionBoundsT& dimensions
)
{
    std::vector< typename DimensionBoundsT::value_type >
        dimension_pos( dimensions.size(), 0 );
    while ( 0 < batches )
    {
        auto dist_it = distributions_first;
        for ( auto pos_it = dimension_pos.begin();
            pos_it != dimension_pos.end();
            ++pos_it
            )
            *pos_it = ( *dist_it++ )( rng );
        ++buffer[ collapse_dimensions( dimension_pos, dimensions ) ];
        --batches;
    }
}


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true
>
std::vector< nodeidx_t >
uniform_distribute_node_counts(
    I num_nodes,
    const tileidx_t& num_bins,
    const RandomManager& rng_manager,
    const bool& balanced,
    const bool& global
)
{
    assert(
        0 < num_bins && 0 < num_nodes // RNG manager initialization is assumed already tested
    );

    // Trivial case
    if ( num_bins == 1 )
        return safe_convert_vector(
            std::vector< I >{ num_nodes }
        );

    // Initialize map and rng distribution
    const auto init_val = balance_node_counts_by_bins(
        num_nodes, num_bins, balanced
    );
    std::vector< I > count_vec( num_bins, init_val );
    if ( num_nodes == 0 )
        return safe_convert_vector( std::move( count_vec ) );

    const auto num_threads = get_max_omp_threads();
    auto [full_batches, partial_batches] =
        batch_node_counts_by_thread_count( num_nodes, num_threads );

    nest::uniform_int_distribution tile_idx_dist;
    tile_idx_dist.param(
        nest::uniform_int_distribution::param_type(
            0, static_cast< nest::uniform_int_distribution::result_type >( num_bins - 1 )
        )
    );

    if ( 0 < full_batches )
    {
        std::vector< std::vector< I > > buffers( num_threads );

#pragma omp taskloop grainsize( 1 ) default( none )\
    shared( rng_manager, buffers )\
    firstprivate( num_threads, num_bins,\
    full_batches, tile_idx_dist, global, balanced )
        for ( vp_t curr_idx = 0; curr_idx < num_threads; ++curr_idx )
        {
            const auto curr_buff = buffers.begin() + curr_idx;
            curr_buff->resize( num_bins, 0 );
            batched_draw(
                *curr_buff,
                full_batches,
                global
                ? rng_manager.get_tid_synced_rng( curr_idx )
                : rng_manager.get_tid_specific_rng( curr_idx ),
                tile_idx_dist,
                num_bins,
                balanced
            );
        }

        left_merge_buffers< I >( buffers, num_threads );
        count_vec.swap( buffers[ 0 ] );
        buffers.clear();

        if ( 0 < init_val )
            for ( auto& nc : count_vec )
                nc += init_val;
    }

    if ( 0 < partial_batches )
        batched_draw(
            count_vec,
            partial_batches,
            global
            ? rng_manager.get_rank_synced_rng()
            : rng_manager.get_rank_specific_rng(),
            tile_idx_dist,
            num_bins,
            balanced
        );

    return safe_convert_vector( std::move( count_vec ) );
}


template < typename I,
    typename std::enable_if_t< std::is_integral_v< rcvref< I > >, bool > = true,
    typename DimensionBoundsT
>
std::vector< nodeidx_t >
uniform_distribute_node_counts(
    I num_nodes,
    const tileidx_t& num_bins,
    const DimensionBoundsT& dimensions,
    const RandomManager& rng_manager,
    const bool& global
)
{
    // Trivial case
    if ( num_bins == 1 )
        return safe_convert_vector(
            std::vector< I >{ num_nodes }
        );

    // Initialize map and rng distribution
    std::vector< I > count_vec( num_bins, 0 );
    if ( num_nodes == 0 )
        return safe_convert_vector( std::move( count_vec ) );

    const auto num_threads = get_max_omp_threads();
    auto [full_batches, partial_batches] =
        batch_node_counts_by_thread_count( num_nodes, num_threads );

    std::vector< nest::uniform_int_distribution > distributions( dimensions.size() );
    auto dim_it = dimensions.begin();
    std::for_each(
        distributions.begin(),
        distributions.end(),
        [ & ]( auto& dist )
        {
            dist.param(
                nest::uniform_int_distribution::param_type(
                    0, static_cast< nest::uniform_int_distribution::result_type >( *dim_it++ - 1 )
                )
            );
        }
    );

    if ( 0 < full_batches )
    {
        std::vector< std::vector< I > > buffers( num_threads );

#pragma omp taskloop grainsize( 1 ) default( none )\
    shared( rng_manager, buffers )\
    firstprivate( num_threads, num_bins,\
    full_batches, distributions, dimensions, global )
        for ( vp_t curr_idx = 0; curr_idx < num_threads; ++curr_idx )
        {
            const auto curr_buff = buffers.begin() + curr_idx;
            curr_buff->resize( num_bins, 0 );
            batched_draw(
                *curr_buff,
                full_batches,
                global
                ? rng_manager.get_tid_synced_rng( curr_idx )
                : rng_manager.get_tid_specific_rng( curr_idx ),
                distributions.begin(),
                dimensions
            );
        }

        left_merge_buffers< I >( buffers, num_threads );
        count_vec.swap( buffers[ 0 ] );
        buffers.clear();
    }

    if ( 0 < partial_batches )
        batched_draw(
            count_vec,
            partial_batches,
            global
            ? rng_manager.get_rank_synced_rng()
            : rng_manager.get_rank_specific_rng(),
            distributions.begin(),
            dimensions
        );

    return safe_convert_vector( std::move( count_vec ) );
}


inline void aggregate_tiled_node_count_by_rank(
    NodeCountVector& node_counts_per_rank,
    TileIdxNodeCountPairListVector& tiled_node_counts_per_rank,
    const tileidx_t& tile_index,
    const nodeidx_t& node_count_in_tile,
    const GridNeighborhood& grid_neighborhood,
    const RandomManager& rng_manager,
    const bool& balanced
)
{
    assert( 0 <= node_count_in_tile );

    if ( node_count_in_tile == 0 )
        return;

    const auto tile_owners = grid_neighborhood.tile_ranks_ownership_map_.cbegin() + tile_index;
    assert( tile_owners != grid_neighborhood.tile_ranks_ownership_map_.end() );
    if ( tile_owners->empty() )
        throw std::runtime_error( "Tile with allocated nodes has no owning rank" );

    assert( tile_owners->size() < std::numeric_limits< tileidx_t >::max() );

    const auto node_counts_per_owning_rank = uniform_distribute_node_counts(
        node_count_in_tile,
        static_cast< tileidx_t >( tile_owners->size() ),
        rng_manager,
        balanced,
        true // global
    );

    auto nc_it = node_counts_per_owning_rank.cbegin();
    for ( const auto& owner_rank : *tile_owners )
    {
        const auto count = *nc_it++;
        if ( count == 0 ) continue;
        node_counts_per_rank[ owner_rank ] += count;
        tiled_node_counts_per_rank[ owner_rank ].emplace_front(
            std::make_pair( tile_index, count )
        );
    }
}


// Here it is assumed that node sequences in each rank are generated externally
// i.e. this library generates a number of nodes per tiles then aggregates by rank
// then another library instantiates the nodes in the rank and returns the sequence
// of nodes generated in the rank
inline DistributedTiledNodeSequenceMap
consolidate_node_sequences_per_tile_per_rank(
    const RankNodeSequenceMap& node_seq_per_rank,
    const TileIdxNodeCountPairListVector& node_counts_per_tile_per_rank
)
{
    nodeidx_t first_node_idx;
    DistributedTiledNodeSequenceMap dist_tns;
    for ( const auto& [rank, node_sequence] : node_seq_per_rank )
    {
        if ( node_sequence.first < 0 ||
            node_sequence.second < 1 )
            throw std::invalid_argument( "Invalid node sequence" );

        first_node_idx = node_sequence.first;
        TileIdxNodeSequenceMap tile_idx_node_sequence_map;
        for ( const auto& [tile_index, node_count]
            : node_counts_per_tile_per_rank.at( rank ) )
        {
            if ( node_count < 1 )
                continue;

            const auto emplace_res = tile_idx_node_sequence_map.emplace(
                std::make_pair(
                    tileidx_t( tile_index ),
                    NodeSequence(
                        first_node_idx,
                        node_count
                    )
                )
            );
            assert( emplace_res.second );

            first_node_idx += node_count;
        }
        assert( ( first_node_idx - node_sequence.first ) == node_sequence.second );

        if ( !tile_idx_node_sequence_map.empty() )
            dist_tns.emplace(
                std::make_pair(
                    vp_t( rank ),
                    std::move( tile_idx_node_sequence_map )
                )
            );
    }

    return dist_tns;
}
}


#endif
