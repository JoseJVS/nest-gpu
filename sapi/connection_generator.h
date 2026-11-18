/*
 *  connection_generator.h
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

#ifndef CONNECTION_GENERATOR_H
#define CONNECTION_GENERATOR_H

#include "connection_containers.h"
#include "connection_rules.h"
#include "dispatch.h"
#include "numeric_functors.h"
#include "sapi_forward_declarations.h"

namespace sapi {
struct ConnectionGenerator {
  bool partition_connections_ = false;
  bool allow_self_connections_ = false;
  bool allow_multiplicity_ = false;
  CONNECTION_RULE rule_ = CONNECTION_RULE::NULL_CM;
  count_t connection_counts_ = 0;
  NFCollection numeric_functors_;

  bool check_parameters() const;

  bool sort_by_pool_indexes() const;

  template <Coordinate CoordT>
  std::size_t
  generate_connections(AnyRNG &rng, ProceduralConnectivityBlocks &proc_block,
                       ConnectionTask<CoordT> &task, const Mask<CoordT> &mask,
                       const bool generate_remote_connections) const;
};

inline bool ConnectionGenerator::check_parameters() const {
  if (!numeric_functors_.weight_functor_.is_initialized() ||
      !numeric_functors_.delay_functor_.is_initialized())
    return false;

  switch (rule_) {
  case CONNECTION_RULE::PAIRWISE_BERNOULLI:
    return numeric_functors_.probability_functor_.is_initialized();

  case CONNECTION_RULE::PAIRWISE_POISSON:
    return numeric_functors_.probability_functor_.is_initialized();

  case CONNECTION_RULE::FIXED_IN_DEGREE:
    return 0 < connection_counts_;

  case CONNECTION_RULE::FIXED_OUT_DEGREE:
    return 0 < connection_counts_;

  default:
    return false;
  }
}

inline bool ConnectionGenerator::sort_by_pool_indexes() const {
  return rule_ == CONNECTION_RULE::FIXED_IN_DEGREE;
}

template <Coordinate CoordT>
std::size_t ConnectionGenerator::generate_connections(
    AnyRNG &rng, ProceduralConnectivityBlocks &proc_block,
    ConnectionTask<CoordT> &task, const Mask<CoordT> &mask,
    const bool generate_remote_connections) const {
  // Pairs across ranks are never self-connections, so remote generation can
  // skip the self-connection check.
  return dispatch_bools(
      [&]<bool allow_self_connections, bool allow_multiplicity>() {
        return sapi::generate_connections<CoordT, allow_self_connections,
                                          allow_multiplicity>(
            rng, proc_block, task, mask, numeric_functors_, connection_counts_,
            rule_);
      },
      generate_remote_connections || allow_self_connections_,
      allow_multiplicity_);
}
} // namespace sapi

#endif
