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

#include "random_manager.h"
#include "rng_creators.h"
#include "vp_interface.h"

namespace sapi {
RandomManager::RandomManager()
    : RandomManager(get_mpi_rank(), get_num_mpi_processes()) {}

RandomManager::RandomManager(const vp_t local_rank, const vp_t num_processes)
    : local_rank_(static_cast<rng_seed_t>(local_rank)),
      num_processes_(static_cast<rng_seed_t>(num_processes)) {
  // Seeds and the rank-paired seed table are indexed by the local rank
  check_rank_in_range(local_rank, num_processes);
  initialize_rng_registry(rng_registry_);
  current_creator_ = rng_registry_.get_creator(DEFAULT_RNG_TYPE_).get();
}

// Re-initialization needs no explicit teardown: the unique_ptr assignments
// below destroy the previous engines and TAArray::clone re-prepares the
// per-thread arrays.
void RandomManager::initialize() {
  rank_paired_seeds_.assign(num_processes_, 0);

  const AnyRNG rng(current_creator_->create());
  rank_synced_rng_ = std::make_unique<AnyRNG>(rng);
  rank_synced_rng_->seed({base_seed_, RANK_SEEDER_});
  rank_specific_rng_ = std::make_unique<AnyRNG>(rng);
  rank_specific_rng_->seed(
      {base_seed_, RANK_SEEDER_, PARITY_SEEDER_, local_rank_});

  tid_synced_rng_vec_.clone(rng);
  tid_specific_rng_vec_.clone(rng);
  tid_rank_paired_rng_vec_.clone(rng);

#pragma omp parallel default(none)                                             \
    shared(tid_synced_rng_vec_, tid_specific_rng_vec_)                         \
    firstprivate(base_seed_, local_rank_)
  {
    const auto tid = get_thread_num();
    tid_synced_rng_vec_.get_thread_item(tid)->seed(
        {base_seed_, THREAD_SEEDER_, static_cast<rng_seed_t>(tid)});
    tid_specific_rng_vec_.get_thread_item(tid)->seed(
        {base_seed_, THREAD_SEEDER_, PARITY_SEEDER_, local_rank_,
         static_cast<rng_seed_t>(tid)});
  }
}
} // namespace sapi
