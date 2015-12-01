#include <cstdlib>

#include <iostream>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <cfloat>
#include <string>
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

using namespace std;
#define NUMBER_OF_ELEMENTS 16
#define NUMBER_OF_CLUSTERS 2
#define MAX_NUMBER_OF_ITERATIONS 20

__device__ __host__
float compute_distance(float x1,float x2,float y1,float y2){
  return sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

__global__
void mapFuncGpu(int * map_data_cluster_index,float *data_x,float *cluster_x,float *data_y,float *cluster_y)
{
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  //for all the points do
  int index=0;
  float minDistance=FLT_MAX;

  for(int i=0;i<NUMBER_OF_CLUSTERS;i++)
  {
    float currDistance=compute_distance(data_x[j],cluster_x[i],data_y[j],cluster_y[i]);
    //printf( "GPU point with index %d has %f distance from cluster %d \n",j,currDistance,i);
    if(currDistance<minDistance)
    {
      minDistance=currDistance;
      index=i;
    }
  }
  //printf( "GPU point with index %d has minimum distance %f from cluster %d \n",j,minDistance,index);
  map_data_cluster_index[j]=index;
}

void reduceFuncGPU(thrust::device_vector<int> &data_cluster_index,thrust::device_vector<float> &data_x,thrust::device_vector<float> &data_y,thrust::device_vector<float> &cendroids_x,thrust::device_vector<float> &cendroids_y)
{
  cout<<"started reduction"<<endl;
  thrust::device_vector<int> d_data_cluster_index_x=data_cluster_index;
	thrust::device_vector<float> cendroid_sumx(NUMBER_OF_CLUSTERS);
	thrust::device_vector<float> cendroid_sumy(NUMBER_OF_CLUSTERS);
	thrust::device_vector<int> new_data_cluster_index;
	thrust::fill(cendroid_sumx.begin(),cendroid_sumx.end(),0);
	thrust::fill(cendroid_sumy.begin(),cendroid_sumy.end(),0);
  cout<<"created and initialized vectors"<<endl;
	thrust::plus<float> binary_op;
	thrust::equal_to<int> binary_pred;
	thrust::device_vector<int> data_cluster_index_y=data_cluster_index;
	cout<<"starting sort"<<endl;
	//sorts data_x and data_y by key
	thrust::sort_by_key(d_data_cluster_index_x.begin(),d_data_cluster_index_x.end(),data_x.begin());
  thrust::sort_by_key(data_cluster_index_y.begin(),data_cluster_index_y.end(),data_y.begin());
	//sums up data_x
  cout<<"starting reduce of x"<<endl;
	thrust::reduce_by_key(d_data_cluster_index.begin(),d_data_cluster_index.end(),data_x.begin(),new_data_cluster_index.begin(),cendroid_sumx.begin(),binary_pred,binary_op);

	//sums up data_y
	cout<<"starting reduce of y"<<endl;
	thrust::reduce_by_key(d_data_cluster_index.begin(),d_data_cluster_index.end(),data_y.begin(),new_data_cluster_index.begin(),cendroid_sumy.begin(),binary_pred,binary_op);
  cout<<"finished reduction"<<endl;
	thrust::device_vector<unsigned int> cluster_begin(NUMBER_OF_ELEMENTS);
  thrust::device_vector<unsigned int> cluster_end(NUMBER_OF_ELEMENTS);
	thrust::counting_iterator<unsigned int>search_begin(0);
  cout<<"started counting"<<endl;
	thrust::lower_bound(d_data_cluster_index.begin(),d_data_cluster_index.end(),search_begin,search_begin+NUMBER_OF_ELEMENTS,cluster_begin.begin());
	thrust::upper_bound(d_data_cluster_index.begin(),d_data_cluster_index.end(),search_begin,search_begin+NUMBER_OF_ELEMENTS,cluster_end.begin());
	thrust::device_vector<int> cluster_count_gpu(NUMBER_OF_CLUSTERS);
	thrust::minus<unsigned int> binary_op2;
	thrust::divides<float> binary_op3;
	thrust::transform(cluster_end.begin(),cluster_end.end(),cluster_begin.begin(),cluster_count_gpu.begin(),binary_op2);
  cout<<"finished counting"<<endl;
	thrust::transform(cendroid_sumx.begin(),cendroid_sumx.end(),cluster_count_gpu.begin(),cendroid_sumx.begin(),binary_op3);
	thrust::transform(cendroid_sumy.begin(),cendroid_sumy.end(),cluster_count_gpu.begin(),cendroid_sumy.begin(),binary_op3);
	cout<<"finished dividing"<<endl;
	cout<<"the count of the first cluster on the gpu is "<<cluster_count_gpu[0]<<endl;
	cout<<"the count of the second cluster on the gpu is "<<cluster_count_gpu[1]<<endl;
	cendroids_x=cendroid_sumx;
	cendroids_y=cendroid_sumy;
}

int main(){
  using namespace thrust;
  cout<<"creating host vectors"<<endl;
  host_vector<float> data_x(NUMBER_OF_ELEMENTS);
  host_vector<float> data_y(NUMBER_OF_ELEMENTS);
  host_vector<int> data_cluster_index(NUMBER_OF_ELEMENTS);
  host_vector<int> initial_centroid_index(NUMBER_OF_CLUSTERS);
  host_vector<float> cendroids_x(NUMBER_OF_CLUSTERS);
  host_vector<float> cendroids_y(NUMBER_OF_CLUSTERS);
  host_vector<float> cendroids_sumx(NUMBER_OF_CLUSTERS);
  host_vector<float> cendroids_sumy(NUMBER_OF_CLUSTERS);
  cout<<"initializing cendroids"<<endl;

  initial_centroid_index[0]=0;
  initial_centroid_index[1]=2;

  cout<<"initializing all points to belong in the sentinel cluster"<<endl;
  //initialize all the points to belong in sentinel cluster -1
  for (int i=0; i<data_cluster_index.size(); i++) {
    data_cluster_index[i]=-1;
  }
  cout<<"initializing the count of all clusters to 0"<<endl;
  //initialize number of points in all cendroids to 0

  for (int i=0; i<cendroids_sumx.size(); i++) {
    cendroids_sumx[i]=0;
    cendroids_sumy[i]=0;
  }

  cout<<"loading data from the text files"<<endl;
  //load data from the two text files
  ifstream is("a.txt");
  ifstream is2("b.txt");
  for (int i=0; i<data_x.size(); i++) {
    is>>data_x[i];
  }

  for (int i=0; i<data_y.size(); i++) {
    is2>>data_y[i];
  }

  cout<<"initializing the data for the two initial cendroids"<<endl;
  //initialize the data for the two initial cendroids
  cendroids_x[0]=data_x[initial_centroid_index[0]];
  cendroids_y[0]=data_y[initial_centroid_index[0]];
  cendroids_x[1]=data_x[initial_centroid_index[1]];
  cendroids_y[1]=data_y[initial_centroid_index[1]];

  //creating and populating device vectors
  cout<<"creating the device vectors"<<endl;
  thrust::device_vector<float> d_data_x(NUMBER_OF_ELEMENTS);
  cout<<"created the data_x vector"<<endl;
  thrust::device_vector<float> d_data_y(NUMBER_OF_ELEMENTS);
  cout<<"created the data vectors"<<endl;

  thrust::device_vector<float> d_cendroids_x(NUMBER_OF_CLUSTERS);
  thrust::device_vector<float> d_cendroids_y(NUMBER_OF_CLUSTERS);

  thrust::device_vector<int> prev_index(NUMBER_OF_ELEMENTS);
  thrust::device_vector<int> d_data_cluster_index(NUMBER_OF_ELEMENTS);

  int * data_cluster_index_ptr=thrust::raw_pointer_cast(&d_data_cluster_index[0]);
  float *map_cluster_x=thrust::raw_pointer_cast(&d_cendroids_x[0]);
  float *map_data_x=thrust::raw_pointer_cast(&d_data_x[0]);
  float *map_cluster_y=thrust::raw_pointer_cast(&d_cendroids_y[0]);
  float *map_data_y=thrust::raw_pointer_cast(&d_data_y[0]);

  bool done=false;
  int numIt=0;
  while(numIt<MAX_NUMBER_OF_ITERATIONS){

    cout<<"calling the map function"<<endl;

    mapFuncGpu<<<NUMBER_OF_ELEMENTS,1>>>(data_cluster_index_ptr,map_data_x,map_cluster_x,map_data_y,map_cluster_y);
    done=thrust::equal(d_data_cluster_index.begin(),d_data_cluster_index.end(),prev_index.begin());
    if(done)break;
    thrust::copy(d_data_cluster_index.begin(),d_data_cluster_index.end(),prev_index.begin());
    cout<<"finished mapping"<<endl;
    reduceFuncGPU(d_data_cluster_index,d_data_x,d_data_y,d_cendroids_x,d_cendroids_y);
    cout<<"finished reduction"<<endl;
    numIt++;
    printf("at iteration %d done is %d \n",numIt,done);
  }

  for(int i=0;i<cendroids_x.size();i++)
  {
    cout<<"the gpu sum_x is "<<d_cendroids_x[i]<<"in the position "<<i<<endl;
    cout<<"the gpu sum_y is "<<d_cendroids_y[i]<<"in the position "<<i<<endl;
  }

  cout<<endl;

  for(int i=0;i<d_data_cluster_index.size();i++)
  {
    cout<<"the element with index "<<i<<" got mapped in the cluster "<<d_data_cluster_index[i]<<endl;
  }
  cout<<endl;
}
