#include <algorithm>
#include <vector>

#include "truncation_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TruncationForward(const int n, const Dtype* in, Dtype* out,
    Dtype lower_bound, Dtype upper_bound) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = min(max(in[index], lower_bound), upper_bound);
  }
}

template <typename Dtype>
void TruncationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype lower_bound = this->layer_param_.truncation_param().lower_bound();
  Dtype upper_bound = this->layer_param_.truncation_param().upper_bound();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TruncationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, lower_bound, upper_bound);
  CUDA_POST_KERNEL_CHECK;
}



INSTANTIATE_LAYER_GPU_FUNCS(TruncationLayer);


}  // namespace caffe
