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

#ifndef AEIFCONDBETAH
#define AEIFCONDBETAH

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "rk5.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

#define MAX_PORT_NUM 20

struct aeif_cond_beta_rk5
{
  int i_node_0_;
};

class aeif_cond_beta : public BaseNeuron
{
 public:
  RungeKutta5<aeif_cond_beta_rk5> rk5_;
  float h_min_;
  float h_;
  aeif_cond_beta_rk5 rk5_data_struct_;
    
  int Init(int i_node_0, int n_neuron, int n_port, int i_group,
	   unsigned long long *seed);

  int Calibrate(float time_min, float time_resolution);
		
  int Update(int it, float t1);
  
  int GetX(int i_neuron, int n_node, float *x) {
    return rk5_.GetX(i_neuron, n_node, x);
  }
  
  int GetY(int i_var, int i_neuron, int n_node, float *y) {
    return rk5_.GetY(i_var, i_neuron, n_node, y);
  }
  
  template<int N_PORT>
    int UpdateNR(int it, float t1);

};

#endif
