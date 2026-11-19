/*
 *  connection_rules.h
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

#ifndef CONNECTION_RULES_H
#define CONNECTION_RULES_H

#include <algorithm>
#include <cassert>
#include <random>
#include <utility>

#include "connection_containers.h"
#include "coordinates.h"
#include "numeric_functors.h"
#include "sapi_forward_declarations.h"

namespace sapi {
// Probability of a candidate. Negative kernel values and NaN are not valid
// probabilities: they count as 0, so the candidate is never connected.
template <Coordinate CoordT>
inline conn_param_t
kernel_probability(const NumericFunctor &kernel,
                   const Displacement<CoordT> &displacement) {
  const auto probability = kernel(displacement);
  return std::isgreaterequal(probability, 0) ? probability : 0;
}

// probability is never negative or NaN (see kernel_probability)
template <typename DistributionT, bool allow_multiplicity>
inline count_t draw_connections(AnyRNG &rng, DistributionT &dist,
                                const conn_param_t probability) {
  if constexpr (std::is_same_v<DistributionT,
                               std::uniform_real_distribution<conn_param_t>>) {
    return std::isless(dist(rng), probability);
  } else {
    // A Poisson mean has to be positive
    if (!std::isless(0, probability))
      return 0;

    dist.param(typename DistributionT::param_type(probability));
    if constexpr (allow_multiplicity) {
      return dist(rng);
    } else {
      return 0 < dist(rng);
    }
  }
}

template <Coordinate CoordT>
inline void check_minimal_displacement(OptDisp<CoordT> &&computed_displacement,
                                       OptDisp<CoordT> &tracked_minimum) {
  if ((!tracked_minimum.first && computed_displacement.first) ||
      (computed_displacement.first &&
       std::isless(computed_displacement.second.distance2_,
                   tracked_minimum.second.distance2_))) {
    tracked_minimum.first = true;
    tracked_minimum.second = computed_displacement.second;
  }
}

// Minimal displacement from driver to pool over the image displacements
// selected by the used_displacements bitmask; first is false if none is in
// the mask.
template <Coordinate CoordT>
inline OptDisp<CoordT>
find_minimal_displacement(const std::pair<nodeidx_t, CoordT> &driver_node,
                          const std::pair<nodeidx_t, CoordT> &pool_node,
                          const Mask<CoordT> &mask,
                          const std::vector<CoordT> &image_displacements,
                          const count_t used_displacements) {
  shift_t index = 0;
  OptDisp<CoordT> min_displacement;
  for (const auto &disp : image_displacements)
    if (0 != (used_displacements & 1 << index++))
      check_minimal_displacement(
          mask.coord_in_mask(driver_node.second, pool_node.second + disp),
          min_displacement);

  return min_displacement;
}

template <Coordinate CoordT, typename DistributionT,
          bool allow_self_connections, bool allow_multiplicity>
count_t compute_minimal_displacement(
    AnyRNG &rng, DistributionT &dist, std::deque<ConnectionInfo> &connections,
    const std::pair<nodeidx_t, CoordT> &driver_node,
    const std::pair<nodeidx_t, CoordT> &pool_node, const Mask<CoordT> &mask,
    const NFCollection &functors,
    const std::vector<CoordT> &image_displacements,
    const count_t used_displacements, const count_t conn_limit) {
  assert(0 < conn_limit);

  if constexpr (!allow_self_connections) {
    if (driver_node.first == pool_node.first)
      return conn_limit;
  }

  const OptDisp<CoordT> min_displacement = find_minimal_displacement(
      driver_node, pool_node, mask, image_displacements, used_displacements);

  if (!min_displacement.first)
    return conn_limit;

  const auto probability = kernel_probability(functors.probability_functor_,
                                              min_displacement.second);

  if constexpr (std::is_same_v<DistributionT,
                               std::poisson_distribution<count_t>>) {
    const count_t multiplicity = std::min(
        conn_limit, draw_connections<DistributionT, allow_multiplicity>(
                        rng, dist, probability));

    if (0 == multiplicity)
      return conn_limit;

    const auto weight = functors.weight_functor_(min_displacement.second);
    const auto delay = functors.delay_functor_(min_displacement.second);
    for (count_t i = 0; i < multiplicity; ++i)
      connections.emplace_back(
          construct_connection_info(pool_node.first, weight, delay));

    return conn_limit - multiplicity;
  } else {
    if ((bool)draw_connections<DistributionT, allow_multiplicity>(
            rng, dist, probability)) {
      connections.emplace_back(construct_connection_info(
          pool_node.first, functors.weight_functor_(min_displacement.second),
          functors.delay_functor_(min_displacement.second)));

      return conn_limit - 1;
    }

    return conn_limit;
  }
}

template <Coordinate CoordT, typename DistributionT,
          bool allow_self_connections, bool allow_multiplicity>
std::size_t generate_probabilistic_connections(
    AnyRNG &rng, DistributionT &dist, ProceduralConnectivityBlocks &proc_block,
    ConnectionTask<CoordT> &task, const Mask<CoordT> &mask,
    const NFCollection &functors, const count_t connection_counts) {
  assert(proc_block.empty() && 0 < task.total_possible_combinations_ &&
         functors.probability_functor_.is_initialized());

  std::size_t tracked_conn_counts = 0;
  const count_t individual_max_conns = 0 < connection_counts
                                           ? connection_counts
                                           : task.total_possible_combinations_;

  proc_block.resize(task.pivot_view_->second.size());
  auto block_it = proc_block.begin();
  for (const auto &driver_view : task.pivot_view_->second) {
    const auto &driver_pair = *driver_view;
    assert(0 <= driver_pair.first);
    block_it->first = static_cast<conn_index_t>(driver_pair.first);
    count_t individual_conn_counts = individual_max_conns;

    for (const auto &combination : task.possible_combinations_) {
      assert(combination.second.image_displacements_ != nullptr);
      const auto &image_displacements =
          *combination.second.image_displacements_;

      for (const auto &pool_view : combination.second.possible_pairs_->second) {
        const auto &pool_pair = *pool_view;

        individual_conn_counts = compute_minimal_displacement<
            CoordT, DistributionT, allow_self_connections, allow_multiplicity>(
            rng, dist, block_it->second, driver_pair, pool_pair, mask, functors,
            image_displacements, combination.second.used_displacements_,
            individual_conn_counts);

        if (individual_conn_counts < 1)
          break;
      }

      if (individual_conn_counts < 1)
        break;
    }

    assert(0 <= individual_conn_counts);

    tracked_conn_counts += individual_max_conns - individual_conn_counts;
    ++block_it;
  }

  task.possible_combinations_.clear();

  return tracked_conn_counts;
}

template <Coordinate CoordT, bool allow_self_connections, bool inverted_pivot,
          bool compute_probability>
void compute_minimal_displacement(
    std::deque<ConnectionInfo> &possible_connections,
    std::deque<conn_param_t> &connection_probabilities,
    count_t &positive_counts, const std::pair<nodeidx_t, CoordT> &driver_node,
    const std::pair<nodeidx_t, CoordT> &pool_node, const Mask<CoordT> &mask,
    const NFCollection &functors,
    const std::vector<CoordT> &image_displacements,
    const count_t used_displacements) {
  if constexpr (!allow_self_connections) {
    if (driver_node.first == pool_node.first)
      return;
  }

  const OptDisp<CoordT> min_displacement = find_minimal_displacement(
      driver_node, pool_node, mask, image_displacements, used_displacements);

  if (!min_displacement.first)
    return;

  if constexpr (inverted_pivot) {
    possible_connections.emplace_back(construct_connection_info(
        driver_node.first, functors.weight_functor_(min_displacement.second),
        functors.delay_functor_(min_displacement.second)));
  } else {
    possible_connections.emplace_back(construct_connection_info(
        pool_node.first, functors.weight_functor_(min_displacement.second),
        functors.delay_functor_(min_displacement.second)));
  }

  if constexpr (compute_probability) {
    const auto probability = kernel_probability(functors.probability_functor_,
                                                min_displacement.second);
    if (std::isless(0, probability))
      ++positive_counts;

    connection_probabilities.emplace_back(probability);
  }
}

template <bool allow_multiplicity, bool has_kernel>
void compute_fixed_connections(
    AnyRNG &rng, std::deque<ConnectionInfo> &selected_connections,
    std::deque<ConnectionInfo> &possible_connections,
    std::deque<conn_param_t> &connection_probabilities,
    const count_t connection_counts) {
  assert(selected_connections.empty());

  // Guaranteed cast from total possible combinations count
  const count_t available_counts =
      static_cast<count_t>(possible_connections.size());

  if (available_counts == 0 || available_counts == connection_counts) {
    selected_connections.swap(possible_connections);
    return;
  }

  if (available_counts < connection_counts) {
    if constexpr (allow_multiplicity) {
      if (available_counts == 1) {
        selected_connections.resize(connection_counts, possible_connections[0]);
        return;
      }
    } else {
      selected_connections.swap(possible_connections);
      return;
    }
  }

  assert(allow_multiplicity || connection_counts < available_counts);
  selected_connections.resize(connection_counts);

  if constexpr (has_kernel) {
    assert(std::cmp_equal(connection_probabilities.size(), available_counts));

    if constexpr (allow_multiplicity) {
      std::discrete_distribution<count_t> dist(connection_probabilities.begin(),
                                               connection_probabilities.end());

      for (count_t index = 0; index < connection_counts; ++index)
        selected_connections[index] = possible_connections[dist(rng)];
    } else {
      // A-ExpJ algorithm from https://doi.org/10.1016/j.ipl.2005.11.003

      std::uniform_real_distribution<conn_param_t> weight_dist(
          std::numeric_limits<conn_param_t>::min(), 1.f);

      count_t index = 0;
      count_t inserted = 0;
      std::multimap<conn_param_t, count_t> priority_map;
      while (inserted < connection_counts && index < available_counts) {
        const auto weight = connection_probabilities[index];
        if (std::isless(0, weight)) {
          priority_map.emplace(std::pow(weight_dist(rng), 1.f / weight), index);
          ++inserted;
        }
        ++index;
      }

      if (0 < inserted && index < available_counts) {
        std::uniform_real_distribution<conn_param_t> jump_dist;
        auto min_key = priority_map.cbegin();
        conn_param_t jump =
            std::log(weight_dist(rng)) / std::log(min_key->first);
        for (; index < available_counts; ++index) {
          const auto weight = connection_probabilities[index];
          if (std::isless(0, weight)) {
            jump -= weight;
            if (std::islessequal(jump, 0)) {
              jump_dist.param(
                  std::uniform_real_distribution<conn_param_t>::param_type(
                      std::pow(min_key->first, weight), 1.f));

              priority_map.erase(min_key);
              priority_map.emplace(std::pow(jump_dist(rng), 1.f / weight),
                                   index);
              min_key = priority_map.cbegin();

              jump = std::log(weight_dist(rng)) / std::log(min_key->first);
            }
          }
        }
      }

      // Only positive candidates reach this point, more than requested
      assert(inserted == connection_counts);
      std::ranges::transform(priority_map, selected_connections.begin(),
                             [&](const auto &entry) {
                               return possible_connections[entry.second];
                             });
    }
  } else {
    assert(connection_probabilities.empty());

    if constexpr (allow_multiplicity) {
      std::uniform_int_distribution<count_t> dist(0, available_counts - 1);

      for (count_t index = 0; index < connection_counts; ++index)
        selected_connections[index] = possible_connections[dist(rng)];
    } else {
      std::sample(possible_connections.begin(), possible_connections.end(),
                  selected_connections.begin(), connection_counts, rng);
    }
  }
}

template <Coordinate CoordT, bool allow_self_connections,
          bool allow_multiplicity, bool inverted_pivot, bool has_kernel>
std::size_t generate_fixed_number_connections(
    AnyRNG &rng, ProceduralConnectivityBlocks &proc_block,
    ConnectionTask<CoordT> &task, const Mask<CoordT> &mask,
    const NFCollection &functors, const count_t connection_counts) {
  assert(proc_block.empty() && 0 < task.total_possible_combinations_ &&
         0 < connection_counts);

  std::deque<ConnectionInfo> possible_connections;
  std::deque<conn_param_t> connection_probabilities;
  count_t positive_counts = 0;

  std::size_t tracked_conn_counts = 0;
  proc_block.resize(task.pivot_view_->second.size());
  auto block_it = proc_block.begin();
  for (const auto &pivot_view : task.pivot_view_->second) {
    const auto &pivot_pair = *pivot_view;
    assert(0 <= pivot_pair.first);
    block_it->first = static_cast<conn_index_t>(pivot_pair.first);

    for (const auto &combination : task.possible_combinations_) {
      assert(combination.second.image_displacements_ != nullptr);
      const auto &image_displacements =
          *combination.second.image_displacements_;

      for (const auto &combination_view :
           combination.second.possible_pairs_->second) {
        const auto &combination_pair = *combination_view;

        if constexpr (inverted_pivot) {
          compute_minimal_displacement<CoordT, allow_self_connections,
                                       inverted_pivot, has_kernel>(
              possible_connections, connection_probabilities, positive_counts,
              combination_pair, pivot_pair, mask, functors, image_displacements,
              combination.second.used_displacements_);
        } else {
          compute_minimal_displacement<CoordT, allow_self_connections,
                                       inverted_pivot, has_kernel>(
              possible_connections, connection_probabilities, positive_counts,
              pivot_pair, combination_pair, mask, functors, image_displacements,
              combination.second.used_displacements_);
        }
      }
    }

    // Candidates with probability 0 are never selected: drop them, keeping
    // the order of the others, before the selection
    if constexpr (has_kernel) {
      if (std::cmp_less(positive_counts, possible_connections.size())) {
        auto conn_keep = possible_connections.begin();
        auto prob_keep = connection_probabilities.begin();
        auto prob_it = connection_probabilities.cbegin();
        for (const auto &conn : possible_connections) {
          if (std::isless(0, *prob_it)) {
            *conn_keep++ = conn;
            *prob_keep++ = *prob_it;
          }
          ++prob_it;
        }
        possible_connections.erase(conn_keep, possible_connections.end());
        connection_probabilities.erase(prob_keep,
                                       connection_probabilities.end());
      }
    }

    compute_fixed_connections<allow_multiplicity, has_kernel>(
        rng, block_it->second, possible_connections, connection_probabilities,
        connection_counts);

    possible_connections.clear();
    connection_probabilities.clear();
    positive_counts = 0;

    tracked_conn_counts += block_it->second.size();
    ++block_it;
  }

  task.possible_combinations_.clear();

  return tracked_conn_counts;
}

template <Coordinate CoordT, bool allow_self_connections,
          bool allow_multiplicity>
std::size_t
generate_connections(AnyRNG &rng, ProceduralConnectivityBlocks &proc_block,
                     ConnectionTask<CoordT> &task, const Mask<CoordT> &mask,
                     const NFCollection &functors,
                     const count_t connection_counts,
                     const CONNECTION_RULE method) {
  switch (method) {
  case CONNECTION_RULE::PAIRWISE_BERNOULLI: {
    std::uniform_real_distribution<conn_param_t> dist(0, 1);
    return generate_probabilistic_connections<
        CoordT, std::uniform_real_distribution<conn_param_t>,
        allow_self_connections, allow_multiplicity>(
        rng, dist, proc_block, task, mask, functors, connection_counts);
  }

  case CONNECTION_RULE::PAIRWISE_POISSON: {
    std::poisson_distribution<count_t> dist;
    return generate_probabilistic_connections<
        CoordT, std::poisson_distribution<count_t>, allow_self_connections,
        allow_multiplicity>(rng, dist, proc_block, task, mask, functors,
                            connection_counts);
  }

  case CONNECTION_RULE::FIXED_IN_DEGREE: {
    if (functors.probability_functor_.is_initialized()) {
      return generate_fixed_number_connections<CoordT, allow_self_connections,
                                               allow_multiplicity, true, true>(
          rng, proc_block, task, mask, functors, connection_counts);
    } else {
      return generate_fixed_number_connections<CoordT, allow_self_connections,
                                               allow_multiplicity, true, false>(
          rng, proc_block, task, mask, functors, connection_counts);
    }
  }

  case CONNECTION_RULE::FIXED_OUT_DEGREE: {
    if (functors.probability_functor_.is_initialized()) {
      return generate_fixed_number_connections<CoordT, allow_self_connections,
                                               allow_multiplicity, false, true>(
          rng, proc_block, task, mask, functors, connection_counts);
    } else {
      return generate_fixed_number_connections<
          CoordT, allow_self_connections, allow_multiplicity, false, false>(
          rng, proc_block, task, mask, functors, connection_counts);
    }
  }

  default:
    assert(false && "Invalid connection method");
    return 0;
  }
}
} // namespace sapi

#endif
