#include <algorithm>
#include <vector>

#include "filler.hpp"
#include "layer.hpp"
#include "neuron_layer.hpp"
#include "arccos_layer.hpp"

namespace caffe {

template <typename Dtype>
void ArccosLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  
  for (int i = 0; i < count; ++i) {
    top_data[i] = acosf(bottom_data[i]);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ArccosLayer);
#endif

INSTANTIATE_CLASS(ArccosLayer);
REGISTER_LAYER_CLASS(Arccos);

}  // namespace caffe
