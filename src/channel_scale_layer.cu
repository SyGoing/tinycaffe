#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "layer.hpp"
#include "math_functions.hpp"
#include "channel_scale_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_scale(const int num, const int channels, const int spatial_dim,
                                     Dtype alpha, const Dtype* data, const Dtype* norm_data,
                                     Dtype beta, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    output_data[index] = alpha * data[index] * norm_data[n * spatial_dim + s] + beta * output_data[index];
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim,
                                   const Dtype* data, Dtype* sum_data) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    sum_data[index] = sum;
  }
}

template <typename Dtype>
void ChannelScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* scale_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  if (do_forward_) {
    if (global_scale_) {
      int count = bottom[0]->count();
      Dtype* scale = this->blobs_[0]->mutable_cpu_data();
      Dtype mean_norm = bottom[1]->asum_data() / (Dtype)bottom[1]->count();
      if (this->phase_ == TRAIN) {
        if (scale[0] < 0) {
          scale[0] = mean_norm;
        }
        else {
          scale[0] = scale[0] * 0.99 + mean_norm * 0.01;
        }
        scale[0] = std::min(scale[0], max_global_scale_);
        scale[0] = std::max(scale[0], min_global_scale_);
      }
      if (top.size() == 2) {
        top[1]->mutable_cpu_data()[0] = scale[0];
      }
      caffe_gpu_scale(count, scale[0], bottom_data, top_data);
    }
    else {
      int num = bottom[0]->num();
      int channels = bottom[0]->channels();
      int spatial_dim = bottom[0]->height() * bottom[0]->width();
      // NOLINT_NEXT_LINE(whitespace/operators)
      kernel_channel_scale<Dtype> << <CAFFE_GET_BLOCKS(num*channels*spatial_dim),
        CAFFE_CUDA_NUM_THREADS >> >(num, channels, spatial_dim, Dtype(1), bottom_data, scale_data, Dtype(0), top_data);
    }
  }
  else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}




INSTANTIATE_LAYER_GPU_FUNCS(ChannelScaleLayer);


}  // namespace caffe