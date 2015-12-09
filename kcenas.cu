#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <functional>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <limits>
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
//#define numberOfPoints 16
#define NUMBER_OF_CLUSTERS 2
#define MAX_NUMBER_OF_ITERATIONS 20

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

int numberOfPoints = 0;

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
  data_x[i] = (curand_normal(&localState));
  data_y[i] = (curand_normal(&localState));
  /* Copy state back to global memory */
  state[i] = localState;
}

vector<string> &split(string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

__device__ __host__
float compute_distance(float x1,float x2,float y1,float y2){
  return sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

__global__
void mapFunction(int * map_data_cluster_index,float *data_x,float *cluster_x,float *data_y,float *cluster_y)
{
  int j=threadIdx.x + blockIdx.x * blockDim.x;
  int index=0;
  float minDistance=FLT_MAX;

  for(int i=0;i<NUMBER_OF_CLUSTERS;i++)
  {
    float currDistance=compute_distance(data_x[j],cluster_x[i],data_y[j],cluster_y[i]);
    if(currDistance<minDistance)
    {
      minDistance=currDistance;
      index=i;
    }
  }
  map_data_cluster_index[j]=index;
}

void reduce(thrust::device_vector<int> &data_cluster_index,thrust::device_vector<float> &data_x,thrust::device_vector<float> &data_y,thrust::device_vector<float> &centroids_x,thrust::device_vector<float> &centroids_y, int numberOfPoints)
{
  thrust::device_vector<int> d_data_cluster_index=data_cluster_index;
	thrust::device_vector<float> centroid_sumx(NUMBER_OF_CLUSTERS);
	thrust::device_vector<float> centroid_sumy(NUMBER_OF_CLUSTERS);
	thrust::device_vector<int> new_data_cluster_index(numberOfPoints);
	thrust::fill(centroid_sumx.begin(),centroid_sumx.end(),0);
	thrust::fill(centroid_sumy.begin(),centroid_sumy.end(),0);
	thrust::plus<float> binary_op;
	thrust::equal_to<int> binary_pred;
	thrust::device_vector<int> data_cluster_index_y=data_cluster_index;
	//sorts data_x and data_y by key (groups the points by cluster, which means that the points belonging to the first cluster appear first in the vector)
	thrust::sort_by_key(d_data_cluster_index.begin(),d_data_cluster_index.end(),data_x.begin());
  thrust::sort_by_key(data_cluster_index_y.begin(),data_cluster_index_y.end(),data_y.begin());
	//sums up data_x
	thrust::reduce_by_key(d_data_cluster_index.begin(),d_data_cluster_index.end(),data_x.begin(),new_data_cluster_index.begin(),centroid_sumx.begin(),binary_pred,binary_op);
	//sums up data_y
	thrust::reduce_by_key(d_data_cluster_index.begin(),d_data_cluster_index.end(),data_y.begin(),new_data_cluster_index.begin(),centroid_sumy.begin(),binary_pred,binary_op);
	thrust::device_vector<unsigned int> cluster_begin(numberOfPoints);
  thrust::device_vector<unsigned int> cluster_end(numberOfPoints);
	thrust::counting_iterator<unsigned int>search_begin(0);
	thrust::lower_bound(d_data_cluster_index.begin(),d_data_cluster_index.end(),search_begin,search_begin+numberOfPoints,cluster_begin.begin());
	thrust::upper_bound(d_data_cluster_index.begin(),d_data_cluster_index.end(),search_begin,search_begin+numberOfPoints,cluster_end.begin());
	thrust::device_vector<int> cluster_count_gpu(NUMBER_OF_CLUSTERS);
	thrust::minus<unsigned int> binary_op2;
	thrust::divides<float> binary_op3;
	thrust::transform(cluster_end.begin(),cluster_end.end(),cluster_begin.begin(),cluster_count_gpu.begin(),binary_op2);
	thrust::transform(centroid_sumx.begin(),centroid_sumx.end(),cluster_count_gpu.begin(),centroid_sumx.begin(),binary_op3);
	thrust::transform(centroid_sumy.begin(),centroid_sumy.end(),cluster_count_gpu.begin(),centroid_sumy.begin(),binary_op3);
	cout << "Number of points in the first cluster is " << cluster_count_gpu[0]<<endl;
	cout << "Number of points in the second cluster is " << cluster_count_gpu[1]<<endl;
	centroids_x=centroid_sumx;
	centroids_y=centroid_sumy;
}

int main() {
  using namespace thrust;
  srand(time(NULL));

  numberOfPoints = rand() % 10000 + 1000;
  curandState *devStates;
  CUDA_CALL(cudaMalloc((void **)&devStates, numberOfPoints * sizeof(curandState)));

  setup_kernel<<<numberOfPoints, 1>>>(devStates);


  printf("Generating random numbers based on a normal distribution\n");

  //CUDA_CALL(cudaMemcpy(hostResults, devResults, numberOfPoints * sizeof(int), cudaMemcpyDeviceToHost));

  host_vector<float> data_x(numberOfPoints);
  host_vector<float> data_y(numberOfPoints);
  host_vector<int> data_cluster_index(numberOfPoints);
  host_vector<float> centroids_x(NUMBER_OF_CLUSTERS);
  host_vector<float> centroids_y(NUMBER_OF_CLUSTERS);
  host_vector<float> centroids_sumx(NUMBER_OF_CLUSTERS);
  host_vector<float> centroids_sumy(NUMBER_OF_CLUSTERS);

  cout << "Initializing the centroids" << endl;

  //initialize all the points to belong in sentinel cluster -1
  for (int i = 0; i < data_cluster_index.size(); i++) {
    data_cluster_index[i]=-1;
  }

  //initialize number of points in all centroids to 0
  for (int i = 0; i < centroids_sumx.size(); i++) {
    centroids_sumx[i]=0;
    centroids_sumy[i]=0;
  }

  cout << "Initializing the data for the initial centroids" << endl;
  centroids_x[0]=0.1;
  centroids_y[0]=0.3;
  centroids_x[1]=0.5;
  centroids_y[1]=0.5;

  //creating and populating device vectors
  thrust::device_vector<float> d_data_x = data_x;
  thrust::device_vector<float> d_data_y = data_y;

  thrust::device_vector<float> d_centroids_x = centroids_x;
  thrust::device_vector<float> d_centroids_y = centroids_y;

  thrust::device_vector<int> prev_index(numberOfPoints);
  thrust::device_vector<int> d_data_cluster_index = data_cluster_index;

  int * data_cluster_index_ptr=thrust::raw_pointer_cast(&d_data_cluster_index[0]);
  float *map_cluster_x=thrust::raw_pointer_cast(&d_centroids_x[0]);
  float *map_data_x=thrust::raw_pointer_cast(&d_data_x[0]);
  float *map_cluster_y=thrust::raw_pointer_cast(&d_centroids_y[0]);
  float *map_data_y=thrust::raw_pointer_cast(&d_data_y[0]);

  generate_normal_kernel<<<numberOfPoints, 1>>>(devStates, map_data_x, map_data_y);

  bool done = false;
  int i = 0;
  while(i < MAX_NUMBER_OF_ITERATIONS) {

    cout << "Calling the map function with iteration number " << i << endl;

    mapFunction<<<numberOfPoints,1>>>(data_cluster_index_ptr,map_data_x,map_cluster_x,map_data_y,map_cluster_y);
    // Check if the corresponding cluster for each point changed
    done = thrust::equal(d_data_cluster_index.begin(),d_data_cluster_index.end(),prev_index.begin());
    if (done) {
      cout << "Clusters for each point remained the same! Terminating..." << endl;
      break;
    } else {
      cout << "Some points changed their corresponding cluster! Will do another iteration!" << endl;
    }
    // Copy this cluster index to another value to compare the next index to it
    thrust::copy(d_data_cluster_index.begin(),d_data_cluster_index.end(),prev_index.begin());
    reduce(d_data_cluster_index,d_data_x,d_data_y,d_centroids_x,d_centroids_y,numberOfPoints);
    i++;
  }

  for(int i=0;i<centroids_x.size();i++)
  {
    cout << "The X axis value of the centroid number " << i << " is " << d_centroids_x[i] << endl;
    cout << "The Y axis value of the centroid number " << i << " is " << d_centroids_y[i] << endl;
  }

  cout << "\n\n\n";
  int n0 = 0;
  int n1 = 0;

  for(int i=0;i<d_data_cluster_index.size();i++)
  {
    if (d_data_cluster_index[i] == 0) {
      n0++;
    } else {
      n1++;
    }
  }

  cout << "Number os points in cluster 0 = " << n0 << endl;
  cout << "Number os points in cluster 1 = " << n1 << endl;
}
