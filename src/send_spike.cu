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

#include <config.h>
#include <stdio.h>
#include "send_spike.h"
#include "cuda_error.h"

int *d_SpikeNum;
int *d_SpikeSourceIdx;
int *d_SpikeConnIdx;
float *d_SpikeHeight;
int *d_SpikeTargetNum;

__device__ int MaxSpikeNum;
__device__ int *SpikeNum;
__device__ int *SpikeSourceIdx;
__device__ int *SpikeConnIdx;
__device__ float *SpikeHeight;
__device__ int *SpikeTargetNum;

__device__ void SendSpike(int i_source, int i_conn, float height,
			  int target_num)
{
  int pos = atomicAdd(SpikeNum, 1);
  SpikeSourceIdx[pos] = i_source;
  SpikeConnIdx[pos] = i_conn;
  SpikeHeight[pos] = height;
  SpikeTargetNum[pos] = target_num;
}

__global__ void DeviceSpikeInit(int *spike_num, int *spike_source_idx,
				int *spike_conn_idx, float *spike_height,
				int *spike_target_num,
				int max_spike_num)
{
  SpikeNum = spike_num;
  SpikeSourceIdx = spike_source_idx;
  SpikeConnIdx = spike_conn_idx;
  SpikeHeight = spike_height;
  SpikeTargetNum = spike_target_num;
  MaxSpikeNum = max_spike_num;
  *SpikeNum = 0;
}


void SpikeInit(int max_spike_num)
{
  //h_SpikeTargetNum = new int[PrefixScan::AllocSize];

  gpuErrchk(cudaMalloc(&d_SpikeNum, sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeSourceIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeConnIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeHeight, max_spike_num*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_SpikeTargetNum, max_spike_num*sizeof(int)));
  //printf("here: SpikeTargetNum size: %d", max_spike_num);
  DeviceSpikeInit<<<1,1>>>(d_SpikeNum, d_SpikeSourceIdx, d_SpikeConnIdx,
			   d_SpikeHeight, d_SpikeTargetNum, max_spike_num);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void SpikeReset()
{
  *SpikeNum = 0;
}
