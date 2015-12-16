#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

int numberOfPoints = 0;
int numberOfClusters = 0;

__global__ void setup_kernel(curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets different seed, a different sequence
       number, no offset */
    curand_init(7+id, id, 0, &state[id]);
}

__global__ void generate_normal_kernel(curandState *state, float *xPoints, float *yPoints) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[i];
  /* Generate pseudo-random uniforms */
  xPoints[i] = curand_normal(&localState);
  yPoints[i] = curand_normal(&localState);
  /* Copy state back to global memory */
  state[i] = localState;
}

__device__ __host__
float compute_distance(float x1,float x2,float y1,float y2) {
  return sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

__global__
void mapFunction(int * map_data_cluster_index, float *xPoints, float *xCentroids, float *yPoints, float *yCentroids, int numberOfClusters) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int index = 0;
  float minDistance = FLT_MAX;

  for(int i = 0; i < numberOfClusters; i++) {
    float currentDistance = compute_distance(xPoints[j],xCentroids[i],yPoints[j],yCentroids[i]);
    if(currentDistance<minDistance)
    {
      minDistance = currentDistance;
      index = i;
    }
  }
  map_data_cluster_index[j] = index;
}

__global__
void reduce (int *clusterIndex, float *xPoints, float *yPoints, float *sumX, float *sumY, int *nElemsX, int *nElemsY) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  int index = clusterIndex[j];

  atomicAdd(&sumX[index],xPoints[j]);
  atomicAdd(&sumY[index],yPoints[j]);
  atomicAdd(&nElemsX[index],1);
  atomicAdd(&nElemsY[index],1);
}

__global__
void calculateNewCentroids (float *xCentroids, float *yCentroids, float *sumX, float *sumY, int *nElemsX, int * nElemsY) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  xCentroids[j] = (float) (sumX[j] / nElemsX[j]);
  yCentroids[j] = (float) (sumY[j] / nElemsY[j]);

  //printf ("Number of points in cluster %d is %d\n",j,nElemsX[j]);
}

int main(int agrc, char **argv) {
  cudaSetDevice(1);
  cudaFree(0);

  srand(time(NULL));

  clock_t tStart = clock();

  numberOfPoints = atoi(argv[1]);

  if (numberOfPoints % 2 == 0) {
    numberOfPoints++;
  }

  int t = 0;

  if (numberOfPoints % 10 != 0) {
    t = 1024;
  } else {
    t = 1000;
  }

  numberOfClusters = atoi(argv[2]);

  int tc = 0;

  if (numberOfClusters % 10 != 0) {
    tc = 64;
  } else {
    tc = 60;
  }

  int maxNumberOfIterations = atoi(argv[3]);

  int n = numberOfPoints / t;
  int nc = numberOfClusters / tc;

  curandState *devStates, *devStates2;
  CUDA_CALL(cudaMalloc((void **)&devStates, n * t * sizeof(curandState)));
  CUDA_CALL(cudaMalloc((void **)&devStates2, nc * tc * sizeof(curandState)));

  setup_kernel<<<n, t>>>(devStates);
  setup_kernel<<<nc, tc>>>(devStates2);

  thrust::host_vector<int> clusterIndex(numberOfPoints);

  //initialize all the points to belong in sentinel cluster -1
  for (int i = 0; i < clusterIndex.size(); i++) {
    clusterIndex[i] = -1;
  }

  //creating and populating device vectors
  thrust::device_vector<float> xPoints(numberOfPoints);
  thrust::device_vector<float> yPoints(numberOfPoints);
  thrust::device_vector<float> xCentroids(numberOfClusters);
  thrust::device_vector<float> yCentroids(numberOfClusters);
  thrust::device_vector<int> previousIndex(numberOfPoints);
  thrust::device_vector<int> deviceClusterIndex = clusterIndex;

  int *clusterIndexPointer = thrust::raw_pointer_cast(&deviceClusterIndex[0]);
  float *xCentroidsPointer = thrust::raw_pointer_cast(&xCentroids[0]);
  float *xPointsPointer = thrust::raw_pointer_cast(&xPoints[0]);
  float *yCentroidsPointer = thrust::raw_pointer_cast(&yCentroids[0]);
  float *yPointsPointer = thrust::raw_pointer_cast(&yPoints[0]);

  generate_normal_kernel<<<n, t>>>(devStates, xPointsPointer, yPointsPointer);
  generate_normal_kernel<<<nc, tc>>>(devStates2, xCentroidsPointer, yCentroidsPointer);

  bool done = false;
  int i = 0;

  while(i < maxNumberOfIterations) {
    float *sumX, *sumY;
    int *nElemsX, *nElemsY;

    CUDA_CALL(cudaMalloc((void **)&sumX, nc * tc * sizeof(float)));
    CUDA_CALL(cudaMemset(sumX, 0, nc * tc *  sizeof(float)));

    CUDA_CALL(cudaMalloc((void **)&sumY, nc * tc * sizeof(float)));
    CUDA_CALL(cudaMemset(sumY, 0, nc * tc *  sizeof(float)));

    CUDA_CALL(cudaMalloc((void **)&nElemsX, nc * tc * sizeof(int)));
    CUDA_CALL(cudaMemset(nElemsX, 0, nc * tc *  sizeof(int)));

    CUDA_CALL(cudaMalloc((void **)&nElemsY, nc * tc * sizeof(int)));
    CUDA_CALL(cudaMemset(nElemsY, 0, nc * tc *  sizeof(int)));

    printf("Calling the map function with iteration number %d\n", i);

    mapFunction<<<n, t>>>(clusterIndexPointer,xPointsPointer,xCentroidsPointer,yPointsPointer,yCentroidsPointer, numberOfClusters);
    // Check if the corresponding cluster for each point changed
    done = thrust::equal(deviceClusterIndex.begin(),deviceClusterIndex.end(),previousIndex.begin());
    if (done) {
      printf("Clusters for each point remained the same! Terminating...\n");
      break;
    } else {
      printf("Some points changed their corresponding cluster! Will do another iteration!\n");
    }
    // Copy this cluster index to another value to compare the next index to it
    thrust::copy(deviceClusterIndex.begin(),deviceClusterIndex.end(),previousIndex.begin());
    reduce<<<n, t>>>(clusterIndexPointer, xPointsPointer, yPointsPointer, sumX, sumY, nElemsX, nElemsY);
    calculateNewCentroids<<<nc,tc>>>(xCentroidsPointer, yCentroidsPointer, sumX, sumY, nElemsX, nElemsY);
    i++;
  }

  /*for(int i = 0; i < xCentroids.size(); i++)
  {
    cout << "The X axis value of the centroid number " << i << " is " << xCentroids[i] << endl;
    cout << "The Y axis value of the centroid number " << i << " is " << yCentroids[i] << endl;
  }*/

  printf("Time taken mapping and reducing: %.5fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
