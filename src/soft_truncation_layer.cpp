#include <algorithm>
#include <vector>

#include "soft_truncation_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftTruncationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype c = this->layer_param_.soft_truncation_param().c();
  for (int i = 0; i < count; ++i) {
    top_data[i] = 1 - exp(bottom_data[i] / (-c));
  }
}



#ifdef CPU_ONLY
STUB_GPU(SoftTruncationLayer);
#endif

INSTANTIATE_CLASS(SoftTruncationLayer);
REGISTER_LAYER_CLASS(SoftTruncation);

}  // namespace caffe
