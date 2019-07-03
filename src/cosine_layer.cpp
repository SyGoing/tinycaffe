#include <algorithm>
#include <vector>

#include "filler.hpp"
#include "layer.hpp"
#include "neuron_layer.hpp"
#include "cosine_layer.hpp"

namespace caffe {

template <typename Dtype>
void CosineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  
  for (int i = 0; i < count; ++i) {
    top_data[i] = cos(bottom_data[i]);
  }
}


#ifdef CPU_ONLY
STUB_GPU(CosineLayer);
#endif

INSTANTIATE_CLASS(CosineLayer);
REGISTER_LAYER_CLASS(Cosine);

}  // namespace caffe
