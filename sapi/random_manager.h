/*
 *  random_manager.h
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

#ifndef RANDOM_MANAGER_H
#define RANDOM_MANAGER_H

#include "creator_registry.h"
#include "thread_aligned_array.h"


namespace sapi
{
class RandomManager
{
public:
    const uint32_t local_rank_;
    const vp_t num_processes_;

    RandomManager();
    RandomManager( const RandomManager& ) = delete;
    RandomManager( RandomManager&& ) = default;
    ~RandomManager() = default;

    RandomManager( const vp_t&, const vp_t& );

    void initialize();

    bool is_initialized() const;

    AnyRNG* get_rank_synced_rng() const;
    AnyRNG* get_rank_specific_rng() const;
    AnyRNG* get_tid_synced_rng( const vp_t& ) const;
    AnyRNG* get_tid_specific_rng( const vp_t& ) const;

    AnyRNG* reseed_rank_paired_rng(
        const vp_t&,
        const vp_t&,
        const bool&,
        const tileidx_t&,
        const tileidx_t&
    ) const;

    void update_rank_paired_seed( const vp_t& );

    void set_rng_seed( const uint32_t& );

    void set_rng_type( const std::string& );

protected:
    bool initialized_ = false;
    vp_t num_threads_;

    CreatorRegistry< AnyRNG > rng_registry_;
    StateLessCreator< AnyRNG >* current_creator_;
    std::string current_rng_type_ = DEFAULT_RNG_TYPE_;

    std::unique_ptr< AnyRNG > rank_synced_rng_;
    std::unique_ptr< AnyRNG > rank_specific_rng_;
    TAArray< AnyRNG > tid_synced_rng_vec_;
    TAArray< AnyRNG > tid_specific_rng_vec_;
    TAArray< AnyRNG > tid_rank_paired_rng_vec_;

    uint32_t base_seed_ = DEFAULT_BASE_SEED_;
    std::vector< uint32_t > rank_paired_seeds_;
};


inline bool RandomManager::is_initialized() const
{
    return initialized_;
}


inline AnyRNG* RandomManager::get_rank_synced_rng() const
{
    assert( initialized_ );
    return rank_synced_rng_.get();
}


inline AnyRNG* RandomManager::get_rank_specific_rng() const
{
    assert( initialized_ );
    return rank_specific_rng_.get();
}


inline AnyRNG* RandomManager::get_tid_synced_rng( const vp_t& tid ) const
{
    assert( initialized_ );
    return tid_synced_rng_vec_.get_thread_item( tid );
}


inline AnyRNG* RandomManager::get_tid_specific_rng( const vp_t& tid ) const
{
    assert( initialized_ );
    return tid_specific_rng_vec_.get_thread_item( tid );
}


inline AnyRNG* RandomManager::reseed_rank_paired_rng(
    const vp_t& tid,
    const vp_t& target_rank,
    const bool& inverted_source_target,
    const tileidx_t& tile_idx,
    const tileidx_t& leaf_idx
) const
{
    assert(
        initialized_ && 0 <= target_rank && target_rank < num_processes_
    );
    auto rng = tid_rank_paired_rng_vec_.get_thread_item( tid );
    rng->seed(
        {
            base_seed_,
            THREAD_SEEDER_,
            PARITY_SEEDER_,
            inverted_source_target ? static_cast< uint32_t >( target_rank ) : local_rank_,
            inverted_source_target ? local_rank_ : static_cast< uint32_t >( target_rank ),
            rank_paired_seeds_[ target_rank ],
            static_cast< uint32_t >( tile_idx ),
            static_cast< uint32_t >( leaf_idx )
        }
    );
    return rng;
}


inline void RandomManager::update_rank_paired_seed(
    const vp_t& target_rank
)
{
    assert( 0 <= target_rank && target_rank < num_processes_ );
#pragma omp atomic
    ++rank_paired_seeds_[ target_rank ];
}


inline void RandomManager::set_rng_seed( const uint32_t& seed )
{
    base_seed_ = seed;
    initialize();
}


inline void RandomManager::set_rng_type( const std::string& name )
{
    current_creator_ = rng_registry_.get_creator( name ).get();
    current_rng_type_ = name;
    initialize();
}
}


#endif
