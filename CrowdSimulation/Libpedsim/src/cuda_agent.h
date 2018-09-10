#include <cuda_runtime.h>
#include <utility> 
#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

//std::pair<int *, int *> setupCUDAAgents(int *, int *, int *, int *);

int * setupCUDAAgents(int *, int *);// , int *, int *);
int * setupCUDABlurredHM(int *, int *);
void callCUDA(float*, float*, float*, float*, float*, float*, int);
//void callUpdateHeatMap(int **, int **, int **, int **, int);
void callUpdateHeatMap(int *, int *, int *, int *);

/*int *l_heatmap[SIZE*SIZE];
int *l_scaled_heatmap, *l_blurred_heatmap;*/