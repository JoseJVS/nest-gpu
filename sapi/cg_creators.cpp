/*
 *  cg_creators.cpp
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

#include "cg_creators.h"
#include "creator_registry.h"
#include "connection_generator.h"
#include "type_erasure_helpers.h"


namespace sapi
{
template < CONNECTION_METHOD method >
struct BaseGCCreator final : public StateLessCreator< ConnectionGenerator >
{
    ConnectionGenerator create() const override
    {
        ConnectionGenerator cg;
        cg.method_ = method;
        return cg;
    }
};


void initialize_cg_registry( CreatorRegistry< ConnectionGenerator >& cgr )
{
    cgr.register_creator< BaseGCCreator< CONNECTION_METHOD::PAIRWISE_BERNOULLI > >(
        CONNECTION_METHOD_NAMES[ uint8_t( CONNECTION_METHOD::PAIRWISE_BERNOULLI ) ]
    );
    cgr.register_creator< BaseGCCreator< CONNECTION_METHOD::PAIRWISE_POISSON > >(
        CONNECTION_METHOD_NAMES[ uint8_t( CONNECTION_METHOD::PAIRWISE_POISSON ) ]
    );
    cgr.register_creator< BaseGCCreator< CONNECTION_METHOD::FIXED_IN_DEGREE > >(
        CONNECTION_METHOD_NAMES[ uint8_t( CONNECTION_METHOD::FIXED_IN_DEGREE ) ]
    );
    cgr.register_creator< BaseGCCreator< CONNECTION_METHOD::FIXED_OUT_DEGREE > >(
        CONNECTION_METHOD_NAMES[ uint8_t( CONNECTION_METHOD::FIXED_OUT_DEGREE ) ]
    );
}
}
