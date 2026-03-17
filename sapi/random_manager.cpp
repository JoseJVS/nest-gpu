/*
 *  random_manager.cpp
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

#include "vp_interface.h"
#include "random_manager.h"

// Third party
#include "randutils.hpp"
#include "pcg_random.hpp"


namespace sapi
{
template < typename RT, typename RNG >
void RNGModel< RT, RNG >::seed( const std::initializer_list< uint32_t >& l )
{
    randutils::auto_seed_256 seq( l );
    engine_.seed( seq );
}


struct MTGenerator final : public StateLessCreator< AnyRNG >
{
    AnyRNG create() const override
    {
        if constexpr ( std::is_same_v< rng_bits_t, uint32_t > )
        {
            return AnyRNG( std::mt19937{} );
        }
        else
        {
            return AnyRNG( std::mt19937_64{} );
        }
    }
};


struct PCGGenerator final : public StateLessCreator< AnyRNG >
{
    AnyRNG create() const override
    {
        if constexpr ( std::is_same_v< rng_bits_t, uint32_t > )
        {
            return AnyRNG( pcg32_k1024_fast{} );
        }
        else
        {
            return AnyRNG( pcg64_k1024_fast{} );
        }
    }
};


inline void initialize_rng_registry( CreatorRegistry< AnyRNG >& acr )
{
    acr.register_creator< MTGenerator >( "MersenneTwister" );
    acr.register_creator< PCGGenerator >( "PCG" );
}


RandomManager::RandomManager()
    : local_rank_( static_cast< uint32_t >( get_mpi_rank() ) )
    , num_processes_( get_num_mpi_processes() )
{
    initialize_rng_registry( rng_registry_ );
    current_creator_ = rng_registry_.get_creator( current_rng_type_ ).get();
    initialize();
}


RandomManager::RandomManager( const vp_t& local_rank, const vp_t& num_processes )
    : local_rank_( static_cast< uint32_t >( local_rank ) )
    , num_processes_( num_processes )
{
    initialize_rng_registry( rng_registry_ );
    current_creator_ = rng_registry_.get_creator( current_rng_type_ ).get();
    initialize();
}


void RandomManager::initialize()
{
    if (
        initialized_ ||
        !rank_paired_seeds_.empty() ||
        rank_synced_rng_ ||
        rank_specific_rng_ ||
        tid_synced_rng_vec_.is_initialized() ||
        tid_specific_rng_vec_.is_initialized() ||
        tid_rank_paired_rng_vec_.is_initialized()
        )
    {
        rank_synced_rng_.release();
        rank_specific_rng_.release();
        tid_synced_rng_vec_.prepare();
        tid_specific_rng_vec_.prepare();
        tid_rank_paired_rng_vec_.prepare();
        rank_paired_seeds_.clear();
        initialized_ = false;
    }

    rank_paired_seeds_.resize( num_processes_, 0 );

    const auto rng = current_creator_->create();
    rank_synced_rng_ = rng.clone();
    rank_synced_rng_->seed(
        { base_seed_, RANK_SEEDER_ }
    );
    rank_specific_rng_ = rng.clone();
    rank_specific_rng_->seed(
        { base_seed_, RANK_SEEDER_, PARITY_SEEDER_, local_rank_ }
    );

    tid_synced_rng_vec_.clone( rng );
    tid_specific_rng_vec_.clone( rng );
    tid_rank_paired_rng_vec_.clone( rng );

#pragma omp parallel default( none )\
shared( tid_synced_rng_vec_, tid_specific_rng_vec_ )\
firstprivate( base_seed_, local_rank_, current_creator_ )
    {
        const uint32_t tid = static_cast< uint32_t >( get_thread_num() );
        tid_synced_rng_vec_.get_thread_item( tid )->seed(
            { base_seed_, THREAD_SEEDER_, tid }
        );
        tid_specific_rng_vec_.get_thread_item( tid )->seed(
            { base_seed_, THREAD_SEEDER_, PARITY_SEEDER_, local_rank_, tid }
        );
    }

    initialized_ = true;
}
}
