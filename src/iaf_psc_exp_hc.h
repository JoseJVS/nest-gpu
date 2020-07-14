/*
Copyright (C) 2020 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// adapted from:
// https://github.com/nest/nest-simulator/blob/master/models/iaf_psc_exp.h


#ifndef IAFPSCEXPHCH
#define IAFPSCEXPHCH

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"


namespace iaf_psc_exp_hc_ns
{
enum ScalVarIndexes {
  i_I_syn = 0,        // postsynaptic current for exc. inputs
  i_V_m_rel,          // membrane potential relative to E_L
  i_refractory_step,  // refractory step counter
  N_SCAL_VAR
};

enum ScalParamIndexes {
  i_I_e = 0,         // External current in pA
  N_SCAL_PARAM
};

 const std::string iaf_psc_exp_hc_scal_var_name[N_SCAL_VAR] = {
  "I_syn",
  "V_m_rel",
  "refractory_step"
};

const std::string iaf_psc_exp_hc_scal_param_name[N_SCAL_PARAM] = {
  "I_e"
};

} // namespace
 

class iaf_psc_exp_hc : public BaseNeuron
{
 public:
  ~iaf_psc_exp_hc();
  
  int Init(int i_node_0, int n_neuron, int n_port, int i_group,
	   unsigned long long *seed);

  int Update(int it, float t1);

  int Free();

};


#endif
