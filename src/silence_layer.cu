#include <vector>

#include "silence_layer.hpp"
#include "math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SilenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Do nothing.
}



INSTANTIATE_LAYER_GPU_FUNCS(SilenceLayer);

}  // namespace caffe
