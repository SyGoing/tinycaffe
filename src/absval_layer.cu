#include <vector>

#include "absval_layer.hpp"
#include "math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsValLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
}

INSTANTIATE_LAYER_GPU_FUNCS(AbsValLayer);


}  // namespace caffe
