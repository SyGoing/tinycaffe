#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "layer.hpp"
#include "math_functions.hpp"
#include "custom_layers.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void FixCoordinate(const int n, Dtype* in_out,
    Dtype min_value, Dtype max_value) {
    CUDA_KERNEL_LOOP(index, n) {
      in_out[index] = (in_out[index] < min_value && in_out[index] > min_value - 1e-4) ? min_value : in_out[index];
      in_out[index] = (in_out[index] > max_value && in_out[index] < (max_value + 1e-4)) ? max_value : in_out[index];
    }
  }

  template <typename Dtype>
  __global__ void TransformerForward(const int num, const int channels,
    const int spatial_dim, const int height, const int width,
    const Dtype* data, Dtype*  CoordinateSource_data,
    Dtype* transformed_data) {
    CUDA_KERNEL_LOOP(index, num * spatial_dim) {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      int h = s / width;
      int w = s % width;
      Dtype x = CoordinateSource_data[n * 2 * spatial_dim + h * width + w] * height / 2 + (Dtype)height / 2;
      Dtype y = CoordinateSource_data[n * 2 * spatial_dim + spatial_dim + h * width + w] * width / 2 + (Dtype)width / 2;
      if (x >= 0 && x <= height - 1 && y >= 0 && y <= width - 1) {
        for (int c = 0; c < channels; c++) {
          for (int xx = floor(x); xx <= ceil(x); xx++) {
            for (int yy = floor(y); yy <= ceil(y); yy++) {
              transformed_data[(((n * channels + c) * height + h) * width) + w] += data[(((n * channels + c) * height + xx) * width) + yy] * (1 - abs(x - xx)) * (1 - abs(y - yy));
            }
          }
        }
      }
    }
  }

  __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
      (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }

 

  template <typename Dtype>
  void TransformerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* theta_data = bottom[1]->gpu_data();
    const Dtype* CoordinateTarget_data = CoordinateTarget.gpu_data();
    Dtype*  CoordinateSource_data = CoordinateSource.mutable_gpu_data();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int spatial_dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    caffe_gpu_set<Dtype>(top[0]->count(), 0, top_data);//why memset cause error?
    for (int n = 0; n < num; n++) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, spatial_dim, 3,
        Dtype(1), theta_data + n * 6, CoordinateTarget_data, Dtype(0), CoordinateSource_data + n * 2 * spatial_dim);
      FixCoordinate<Dtype> << <CAFFE_GET_BLOCKS(spatial_dim), CAFFE_CUDA_NUM_THREADS >> >(
        spatial_dim, CoordinateSource_data + n * 2 * spatial_dim, -1, 1 - 2 / (Dtype)bottom[0]->shape(2));//height = 10, then max = 9/5-1=0.8
      FixCoordinate<Dtype> << <CAFFE_GET_BLOCKS(spatial_dim), CAFFE_CUDA_NUM_THREADS >> >(
        spatial_dim, CoordinateSource_data + n * 2 * spatial_dim + spatial_dim, -1, 1 - 2 / (Dtype)bottom[0]->shape(3));
    }
    TransformerForward<Dtype> << <CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim,
      bottom[0]->shape(2), bottom[0]->shape(3),
      bottom_data, CoordinateSource_data, top_data);
  }



  INSTANTIATE_LAYER_GPU_FUNCS(TransformerLayer);


}  // namespace caffe
