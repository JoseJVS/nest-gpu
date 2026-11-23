/*
 *  vp_interface.h
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

#ifndef VP_INTERFACE_HPP
#define VP_INTERFACE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sapi_config.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_OMP
#include <omp.h>
#endif

namespace sapi {
// Throws unless 0 <= rank < num_processes; for the classes that use both
inline void check_rank_in_range(const vp_t rank, const vp_t num_processes) {
  if (rank < 0 || num_processes <= rank)
    throw std::invalid_argument("Invalid rank " + std::to_string(rank) +
                                " for " + std::to_string(num_processes) +
                                " processes");
}

inline void init_mpi([[maybe_unused]] vp_t *const argc,
                     [[maybe_unused]] char ***const argv) {
#ifdef HAVE_MPI
  vp_t initialized;
  MPI_Initialized(&initialized);
  if (!initialized) {
    vp_t provided_thread_level;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided_thread_level);
  }
#endif
}

inline void finalize_mpi() {
#ifdef HAVE_MPI
  vp_t finalized;
  MPI_Finalized(&finalized);
  if (!finalized)
    MPI_Finalize();
#endif
}

inline void mpi_barrier() {
#ifdef HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

inline vp_t get_mpi_rank() {
#ifdef HAVE_MPI
  vp_t rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return static_cast<vp_t>(rank);
#else
  return 0;
#endif
}

inline vp_t get_num_mpi_processes() {
#ifdef HAVE_MPI
  vp_t size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return static_cast<vp_t>(size);
#else
  return 1;
#endif
}

inline void init_omp(vp_t num_threads = 0) {
#ifdef HAVE_OMP
  if (omp_in_parallel())
    throw std::runtime_error("Cannot initialize OpenMP in parallel context");
  omp_set_dynamic(false);
  if (num_threads == 0)
    num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
#endif
}

inline vp_t get_max_omp_threads() {
#ifdef HAVE_OMP
  return static_cast<vp_t>(omp_get_max_threads());
#else
  return 1;
#endif
}

inline void set_max_omp_threads(const vp_t num_threads) {
#ifdef HAVE_OMP
  if (omp_in_parallel())
    throw std::runtime_error(
        "Cannot change OpenMP threads in parallel context");
  if (num_threads < 1 || 1 << 30 < num_threads)
    throw std::invalid_argument("Incorrect number of threads to set");
  omp_set_num_threads(num_threads);
#endif
}

inline vp_t get_thread_num() {
#ifdef HAVE_OMP
  return static_cast<vp_t>(omp_get_thread_num());
#else
  return 0;
#endif
}

// Weight a chunk of jobs reaches before the next chunk starts, so that
// total_weight splits into about num_chunks chunks
inline nodeidx_t chunk_target(const nodeidx_t total_weight,
                              const nodeidx_t num_chunks) {
  assert(0 <= total_weight && 0 < num_chunks);
  return std::max<nodeidx_t>(1, total_weight / num_chunks +
                                    (total_weight % num_chunks != 0));
}

// Groups the jobs [0, weights.size()) into consecutive ranges of similar
// total weight: each range but the last reaches ceil(total / num_chunks), so
// there are about num_chunks ranges (one more if trailing jobs weigh 0), and
// a job heavier than that ends a range on its own. Returns the range bounds,
// from 0 to weights.size(). Jobs and weights are local to the rank, so they
// fit in nodeidx_t.
inline std::vector<nodeidx_t>
chunk_bounds(const std::vector<nodeidx_t> &weights,
             const nodeidx_t num_chunks) {
  assert(std::cmp_less(weights.size(), std::numeric_limits<nodeidx_t>::max()));
  const auto num_jobs = static_cast<nodeidx_t>(weights.size());

  nodeidx_t total_weight = 0;
  for (const auto weight : weights) {
    assert(0 <= weight &&
           weight <= std::numeric_limits<nodeidx_t>::max() - total_weight);
    total_weight += weight;
  }
  const auto target = chunk_target(total_weight, num_chunks);

  std::vector<nodeidx_t> bounds{0};
  nodeidx_t chunk_weight = 0;
  nodeidx_t job_end = 0;
  for (const auto weight : weights) {
    chunk_weight += weight;
    ++job_end;
    if (target <= chunk_weight && job_end < num_jobs) {
      bounds.push_back(job_end);
      chunk_weight = 0;
    }
  }
  bounds.push_back(num_jobs);

  return bounds;
}

// Number of chunk tasks for the current thread team
inline nodeidx_t get_num_chunks() {
  const auto num_threads = get_max_omp_threads();
  assert(0 < num_threads &&
         num_threads <=
             std::numeric_limits<nodeidx_t>::max() / CHUNKS_PER_THREAD);
  return CHUNKS_PER_THREAD * num_threads;
}

template <typename T> struct OmpLock {
#ifdef HAVE_OMP
  omp_lock_t lock_{};
#endif

  OmpLock() noexcept {
#ifdef HAVE_OMP
    omp_init_lock(&lock_);
#endif
  }

  OmpLock(const OmpLock &) = delete;
  // omp_lock_t must not be copied bitwise: both copies would be destroyed.
  // A moved-to lock is a fresh, unlocked lock; locks are never moved while
  // held.
  OmpLock(OmpLock &&) noexcept : OmpLock() {}

  ~OmpLock() noexcept {
#ifdef HAVE_OMP
    omp_destroy_lock(&lock_);
#endif
  }

  OmpLock &operator=(const OmpLock &) = delete;
  OmpLock &operator=(OmpLock &&) = delete;

  void lock() noexcept;
  bool try_lock() noexcept;
  void unlock() noexcept;
};

template <typename T> inline void OmpLock<T>::lock() noexcept {
#ifdef HAVE_OMP
  omp_set_lock(&lock_);
#endif
}

template <typename T> inline bool OmpLock<T>::try_lock() noexcept {
#ifdef HAVE_OMP
  return omp_test_lock(&lock_) != 0;
#else
  return true;
#endif
}

template <typename T> inline void OmpLock<T>::unlock() noexcept {
#ifdef HAVE_OMP
  omp_unset_lock(&lock_);
#endif
}

template <typename T> struct OmpLockGuard {
  OmpLock<T> &lock_;

  explicit OmpLockGuard(OmpLock<T> &l) noexcept : lock_(l) { lock_.lock(); }

  OmpLockGuard(const OmpLockGuard &) = delete;
  OmpLockGuard(OmpLockGuard &&) = delete;

  OmpLockGuard &operator=(const OmpLockGuard &) = delete;
  OmpLockGuard &operator=(OmpLockGuard &&) = delete;

  ~OmpLockGuard() noexcept { lock_.unlock(); }
};
} // namespace sapi

#endif
