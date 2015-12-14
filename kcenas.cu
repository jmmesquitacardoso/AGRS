#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;
#define PI 3.14159265359
#define MAX_NUMBER_OF_ITERATIONS 20

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

__global__ void generate_normal_kernel(curandState *state, float *data_x, float *data_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[i];
  /* Generate pseudo-random uniforms */
  data_x[i] = curand_normal(&localState);
  data_y[i] = curand_normal(&localState);
  /* Copy state back to global memory */
  state[i] = localState;
}

__device__ __host__
float compute_distance(float x1,float x2,float y1,float y2) {
  return sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

__global__
void mapFunction(int * map_data_cluster_index, float *data_x, float *cluster_x, float *data_y, float *cluster_y, int numberOfClusters) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int index = 0;
  float minDistance = FLT_MAX;

  for(int i = 0; i < numberOfClusters; i++) {
    float currentDistance = compute_distance(data_x[j],cluster_x[i],data_y[j],cluster_y[i]);
    if(currentDistance<minDistance)
    {
      minDistance = currentDistance;
      index = i;
    }
  }
  map_data_cluster_index[j] = index;
}

__global__
void reduce (int *data_cluster_index, float *data_x, float *data_y, float *centroids_x, float *centroids_y, float *sumX, float *sumY, int *nElemsX, int *nElemsY) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;

  int clusterIndex = data_cluster_index[j];

  sumX[clusterIndex] += data_x[j];
  sumY[clusterIndex] += data_y[j];
  nElemsX[clusterIndex]++;
  nElemsY[clusterIndex]++;
}

int main() {
  srand(time(NULL));
  clock_t tStart = clock();

  numberOfPoints = 128;

  if (numberOfPoints % 2 == 0) {
    numberOfPoints++;
  }

  numberOfClusters = rand() % 8 + 2;

  int n = numberOfPoints / 128;

  curandState *devStates, *devStates2;
  CUDA_CALL(cudaMalloc((void **)&devStates, n * 128 * sizeof(curandState)));
  CUDA_CALL(cudaMalloc((void **)&devStates2, numberOfClusters * sizeof(curandState)));

  setup_kernel<<<n, 128>>>(devStates);
  setup_kernel<<<numberOfClusters, 1>>>(devStates2);

  thrust::host_vector<int> data_cluster_index(numberOfPoints);

  //initialize all the points to belong in sentinel cluster -1
  for (int i = 0; i < data_cluster_index.size(); i++) {
    data_cluster_index[i]=-1;
  }

  //creating and populating device vectors
  thrust::device_vector<float> d_data_x(numberOfPoints);
  thrust::device_vector<float> d_data_y(numberOfPoints);
  thrust::device_vector<float> d_centroids_x(numberOfClusters);
  thrust::device_vector<float> d_centroids_y(numberOfClusters);
  thrust::device_vector<int> prev_index(numberOfPoints);
  thrust::device_vector<int> d_data_cluster_index = data_cluster_index;

  int * data_cluster_index_ptr = thrust::raw_pointer_cast(&d_data_cluster_index[0]);
  float *map_cluster_x = thrust::raw_pointer_cast(&d_centroids_x[0]);
  float *map_data_x = thrust::raw_pointer_cast(&d_data_x[0]);
  float *map_cluster_y = thrust::raw_pointer_cast(&d_centroids_y[0]);
  float *map_data_y = thrust::raw_pointer_cast(&d_data_y[0]);

  generate_normal_kernel<<<n, 128>>>(devStates, map_data_x, map_data_y);
  generate_normal_kernel<<<numberOfClusters, 1>>>(devStates2, map_cluster_x, map_cluster_y);

  bool done = false;
  int i = 0;

  while(i < MAX_NUMBER_OF_ITERATIONS) {
    float *sumX, *sumY, *hostSumX, *hostSumY;
    int *nElemsX, *nElemsY, *hostNElemsX, *hostNElemsY;

    CUDA_CALL(cudaMalloc((void **)&sumX, numberOfClusters * sizeof(float)));
    CUDA_CALL(cudaMemset(sumX, 0, numberOfClusters *  sizeof(float)));

    CUDA_CALL(cudaMalloc((void **)&sumY, numberOfClusters * sizeof(float)));
    CUDA_CALL(cudaMemset(sumY, 0, numberOfClusters *  sizeof(float)));

    hostSumX = (float *)calloc(numberOfClusters, sizeof(float));
    hostSumY = (float *)calloc(numberOfClusters, sizeof(float));

    CUDA_CALL(cudaMalloc((void **)&nElemsX, numberOfClusters * sizeof(int)));
    CUDA_CALL(cudaMemset(nElemsX, 0, numberOfClusters *  sizeof(int)));

    CUDA_CALL(cudaMalloc((void **)&nElemsY, numberOfClusters * sizeof(int)));
    CUDA_CALL(cudaMemset(nElemsX, 0, numberOfClusters *  sizeof(int)));

    hostNElemsX = (int *)calloc(numberOfClusters, sizeof(int));
    hostNElemsY = (int *)calloc(numberOfClusters, sizeof(int));

    printf("Calling the map function with iteration number %d\n", i);

    mapFunction<<<n, 128>>>(data_cluster_index_ptr,map_data_x,map_cluster_x,map_data_y,map_cluster_y, numberOfClusters);
    // Check if the corresponding cluster for each point changed
    done = thrust::equal(d_data_cluster_index.begin(),d_data_cluster_index.end(),prev_index.begin());
    if (done) {
      printf("Clusters for each point remained the same! Terminating...\n");
      break;
    } else {
      printf("Some points changed their corresponding cluster! Will do another iteration!\n");
    }
    // Copy this cluster index to another value to compare the next index to it
    thrust::copy(d_data_cluster_index.begin(),d_data_cluster_index.end(),prev_index.begin());
    //reduceFunction(d_data_cluster_index,d_data_x,d_data_y,d_centroids_x,d_centroids_y,numberOfPoints);
    reduce<<<n, 128>>>(data_cluster_index_ptr, map_data_x, map_data_y, map_cluster_x, map_cluster_y, sumX, sumY, nElemsX, nElemsY);

    CUDA_CALL(cudaMemcpy(hostSumX, sumX, numberOfClusters * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hostSumY, sumY, numberOfClusters * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hostNElemsX, nElemsX, numberOfClusters * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hostNElemsY, nElemsY, numberOfClusters * sizeof(int), cudaMemcpyDeviceToHost));

    for (int j = 0; j < numberOfClusters; j++) {
      d_centroids_x[j] = (float) (hostSumX[j] / hostNElemsX[j]);
      d_centroids_y[j] = (float) (hostSumY[j] / hostNElemsY[j]);
      printf("Host X = %d and Host Y = %d\n",hostNElemsX[j], hostNElemsY[j]);
      printf("Number of points in cluster %d is %d\n",j,hostNElemsX[j]);
    }

    i++;
  }

  for(int i = 0; i < d_centroids_x.size(); i++)
  {
    cout << "The X axis value of the centroid number " << i << " is " << d_centroids_x[i] << endl;
    cout << "The Y axis value of the centroid number " << i << " is " << d_centroids_y[i] << endl;
  }

  printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
