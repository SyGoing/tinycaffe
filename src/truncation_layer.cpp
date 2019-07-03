#include <algorithm>
#include <vector>

#include "truncation_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void TruncationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    Dtype lower_bound = this->layer_param_.truncation_param().lower_bound();
    Dtype upper_bound = this->layer_param_.truncation_param().upper_bound();
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::min(std::max(bottom_data[i], lower_bound), upper_bound);
    }
  }


#ifdef CPU_ONLY
  STUB_GPU(TruncationLayer);
#endif

  INSTANTIATE_CLASS(TruncationLayer);
  REGISTER_LAYER_CLASS(Truncation);

}  // namespace caffe
