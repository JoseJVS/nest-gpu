/*
 *  task_helpers.h
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

#ifndef TASK_HELPERS_H
#define TASK_HELPERS_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "sapi_config.h"
#include "vp_interface.h"

// Helpers that distribute jobs of known weight over OpenMP tasks. The callers
// keep every job's own seeding and output slot, so the grouping never changes
// the output.
namespace sapi {
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

// Assigns the jobs [0, weights.size()) to num_bins bins of similar total
// weight (longest processing time first: the heaviest remaining job goes to
// the lightest bin). Returns the job indices of each bin.
inline std::vector<std::vector<std::size_t>>
balance_into_bins(const std::vector<std::size_t> &weights,
                  const std::size_t num_bins) {
  assert(0 < num_bins && num_bins <= weights.size());

  // (weight, job index), heaviest first
  std::vector<std::pair<std::size_t, std::size_t>> weighted_jobs;
  weighted_jobs.reserve(weights.size());
  for (std::size_t job_index = 0; job_index < weights.size(); ++job_index)
    weighted_jobs.emplace_back(weights[job_index], job_index);
  std::ranges::sort(weighted_jobs, std::greater{});

  // (total weight, bin index), lightest bin on top
  using BinLoad = std::pair<std::size_t, std::size_t>;
  std::priority_queue<BinLoad, std::vector<BinLoad>, std::greater<>> bin_loads;
  for (std::size_t bin = 0; bin < num_bins; ++bin)
    bin_loads.emplace(0, bin);

  std::vector<std::vector<std::size_t>> bins(num_bins);
  for (const auto &[weight, job_index] : weighted_jobs) {
    const auto [load, bin] = bin_loads.top();
    bin_loads.pop();
    bins[bin].push_back(job_index);
    bin_loads.emplace(load + weight, bin);
  }

  return bins;
}
} // namespace sapi

#endif
