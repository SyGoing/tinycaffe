#include <algorithm>
#include <vector>

#include "neuron_layer.hpp"
#include "cosine_layer.hpp"

namespace caffe {

  // CUDA kernele for forward
  template <typename Dtype>
  __global__ void CosineForward(const int n, const Dtype* in, Dtype* out) {
    CUDA_KERNEL_LOOP(index, n) {
      out[index] = cos(in[index]);
    }
  }

  // CUDA kernel for bottom backward
  template <typename Dtype>
  __global__ void CosineBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff) {
    CUDA_KERNEL_LOOP(index, n) {
      out_diff[index] = in_diff[index] * -1 * sin(in_data[index]);
    }
  }

  template <typename Dtype>
  void CosineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();

    // NOLINT_NEXT_LINE(whitespace/operators)
    CosineForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, bottom_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(CosineLayer);


}  // namespace caffe
