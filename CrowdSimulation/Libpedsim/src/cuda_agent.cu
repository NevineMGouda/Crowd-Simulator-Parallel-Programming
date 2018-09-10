#include "cuda_agent.h"
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <chrono>
int **d_agentsDesiredPos;

//NEW
int **d_heatmap;
int *d_heatmapValues;
int **d_scaled_heatmap;
int *d_scaled_heatmapValues;

__shared__ int **d_blurred_heatmap;


int *d_blurred_heatmapValues;


//timing stuff
cudaEvent_t start, stop;
float intensifyTime = 0;
float blurrTime = 0;


//NEW in parameters
__global__
void agentsSetup(int *d_agentsDesiredPosStorageValues, int **d_agentsDesiredPos) {
	int index = threadIdx.x;
	d_agentsDesiredPos[index] = d_agentsDesiredPosStorageValues + SIZE*index;
}


__global__
void blurredHMSetup(int *d_scaled_heatmapValues, int **d_scaled_heatmap, int *d_blurred_heatmapValues, int **d_blurred_heatmap) {
	int index = threadIdx.x;

	for (int i = 0; i < CELLSIZE; i++) {
		d_scaled_heatmap[index*CELLSIZE + i] = d_scaled_heatmapValues + SCALED_SIZE*(index*CELLSIZE + i);
		d_blurred_heatmap[index*CELLSIZE + i] = d_blurred_heatmapValues + SCALED_SIZE*(index*CELLSIZE + i);
		}


	//NEW
	//Question TODO: check if its necessaryy or redudant and if redudant then just remove the loop
	//for (int i = 0; i < SIZE; i++){ d_heatmap[index][i] = 0; }
	//Question TODO: If not redudant then make it more effiecient to include these 2 loops together because this will be very slow and redudant
	for (int i = 0; i < SCALED_SIZE; i++){
		for (int j = 0; j < CELLSIZE; j++) {
			d_scaled_heatmap[index*CELLSIZE + j][i] = 0;
			d_blurred_heatmap[index*CELLSIZE + j][i] = 0;
//			d_scaled_heatmap[index+j][i] = 0;

		}
	}
}

__global__
void HMSetup(int *d_heatmapValues, int **d_heatmap){
	int index = threadIdx.x;
	d_heatmap[index] = d_heatmapValues + SIZE*index;
	for (int i = 0; i < SIZE; i++){ d_heatmap[index][i] = 0; }
}

int *setupCUDAAgents(int *d_agentsDesiredPosStorageValues, int *agentsDesiredPosStorageValues) {




	cudaMalloc(&d_agentsDesiredPosStorageValues, SIZE*SIZE*sizeof(int));
	cudaMalloc(&d_agentsDesiredPos, SIZE*sizeof(int *));

	agentsSetup << <1, SIZE >> >(d_agentsDesiredPosStorageValues, d_agentsDesiredPos);

	return d_agentsDesiredPosStorageValues;
}

int *setupCUDABlurredHM(int * d_blurred_heatmapValues, int * blurred_heatmapValues) {

	cudaMalloc(&d_heatmapValues, SIZE*SIZE*sizeof(int));
	cudaMalloc(&d_heatmap, SIZE*sizeof(int *));
	cudaMalloc(&d_scaled_heatmapValues, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMalloc(&d_scaled_heatmap, SCALED_SIZE*sizeof(int *));
	cudaMalloc(&d_blurred_heatmapValues, SCALED_SIZE*SCALED_SIZE*sizeof(int));
	cudaMalloc(&d_blurred_heatmap, SCALED_SIZE*sizeof(int *));

	cudaMemcpy(d_blurred_heatmapValues, blurred_heatmapValues, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyHostToDevice);
	blurredHMSetup << <1, SIZE >> >(d_scaled_heatmapValues, d_scaled_heatmap, d_blurred_heatmapValues, d_blurred_heatmap);
	HMSetup << <1, SIZE >> >(d_heatmapValues, d_heatmap);
	return d_blurred_heatmapValues;

}



__global__
void computeNextDesiredPositionCUDA(float* d_agentsX, float* d_agentsY, float* d_agentsDestX, float* d_agentsDestY, float* d_agentsDestR, float* d_agentsDestReached) {

	bool agentReachedDestination = false;

	double diffX = d_agentsDestX[threadIdx.x] - d_agentsX[threadIdx.x];
	double diffY = d_agentsDestY[threadIdx.x] - d_agentsY[threadIdx.x];
	double length = sqrt(diffX * diffX + diffY * diffY);

	d_agentsX[threadIdx.x] = (int)round(d_agentsX[threadIdx.x] + diffX / length);
	d_agentsY[threadIdx.x] = (int)round(d_agentsY[threadIdx.x] + diffY / length);

	diffX = d_agentsDestX[threadIdx.x] - d_agentsX[threadIdx.x];
	diffY = d_agentsDestY[threadIdx.x] - d_agentsY[threadIdx.x];
	length = sqrt(diffX * diffX + diffY * diffY);

	agentReachedDestination = length < d_agentsDestR[threadIdx.x];

	if (agentReachedDestination) { d_agentsDestReached[threadIdx.x] = 1; }
}
void callCUDA(float* d_agentsX, float* d_agentsY, float* d_agentsDestX, float* d_agentsDestY, float* d_agentsDestR, float* d_agentsDestReached, int N) {

	computeNextDesiredPositionCUDA << <1, N >> >(d_agentsX, d_agentsY, d_agentsDestX, d_agentsDestY, d_agentsDestR, d_agentsDestReached);

}

__global__
void intensifyHeat(int **d_agentsDesiredPos, int **d_heatmap, int ** d_scaled_heatmap){

	int index1 = threadIdx.x;
	int index2 = blockIdx.x;

	// fadeheat
	d_heatmap[index1][index2] = (int)round(d_heatmap[index1][index2] * 0.80);

	if (d_agentsDesiredPos[index1][index2] > 0){
		d_heatmap[index1][index2] = d_agentsDesiredPos[index1][index2] * 140;
		if (d_heatmap[index1][index2] > 255) { d_heatmap[index1][index2] = 255; }
	}
	int val = d_heatmap[index1][index2];

	// Scale the data for visual representation
	for (int cellY = 0; cellY < CELLSIZE; cellY++){
		for (int cellX = 0; cellX < CELLSIZE; cellX++){
			d_scaled_heatmap[index1 * CELLSIZE + cellY][index2 * CELLSIZE + cellX] = val;
		}
	}
}


__global__
void blurrHeatMapBlock(int **d_scaled_heatmap, int **d_blurred_heatmap) {
	int startrow = blockIdx.x;
	int startcol = threadIdx.x;
	__shared__ int d_scaled_heatmap_block[5][1000];

	if (!(startrow < 2 || startrow > 990))
	{
		for (int row = -2; row < 3; row++)
		{
			d_scaled_heatmap_block[row+2][startcol] = d_scaled_heatmap[startrow + row][startcol];
		}
	}
	__syncthreads();

	int w[5][5] = {
		{ 1, 4, 7, 4, 1 },
		{ 4, 16, 26, 16, 4 },
		{ 7, 26, 41, 26, 7 },
		{ 4, 16, 26, 16, 4 },
		{ 1, 4, 7, 4, 1 }
	};

	#define WEIGHTSUM 273

	if (startrow < 3 || startcol < 3 || startrow > 990 || startcol > 990) return;

			int sum = 0;

				for (int k = -2; k < 3; k++) {
					for (int l = -2; l < 3; l++) {
						sum += w[2 + k][2 + l]*d_scaled_heatmap_block[k+2][startcol + l + 2];
					}
				}
				int value = sum / WEIGHTSUM;
				d_blurred_heatmap[startrow][startcol] = 0x0000FF00 | value << 24;

}


void callUpdateHeatMap(int *d_agentsDesiredPosStorageValues, int *agentsDesiredPosStorageValues, int *d_blurred_heatmapValues, int *blurred_heatmapValues){

	// timing stuff, new
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// copy values to the device
	cudaMemcpyAsync(d_agentsDesiredPosStorageValues, agentsDesiredPosStorageValues, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
	intensifyHeat << < SIZE, SIZE >> >(d_agentsDesiredPos, d_heatmap, d_scaled_heatmap);                    

	blurrHeatMapBlock << < 1000, 1000 >> >(d_scaled_heatmap, d_blurred_heatmap);

	cudaMemcpyAsync(blurred_heatmapValues, d_blurred_heatmapValues, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

}
