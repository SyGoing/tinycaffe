#include <algorithm>
#include <vector>

#include "layer.hpp"
#include "math_functions.hpp"
#include "custom_layers.hpp"

namespace caffe {

  template <typename Dtype>
  void GramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->channels(), 1);
  }

  template <typename Dtype>
  void GramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int num = bottom[0]->num();
    int channel = bottom[0]->channels();
    int spatial_dim = bottom[0]->height() * bottom[0]->width();
    Blob<Dtype> temp;
    temp.ReshapeLike(*bottom[0]);
    caffe_copy<Dtype>(bottom[0]->count(), bottom_data, temp.mutable_cpu_data());

    for (int n = 0; n < num; n++) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel, channel, spatial_dim,
        1 / (Dtype)spatial_dim / (Dtype)channel, bottom_data + n * spatial_dim * channel, temp.cpu_data() + n * spatial_dim * channel, Dtype(0), top_data + n * channel * channel);
    }
  }

  

#ifdef CPU_ONLY
  STUB_GPU(GramLayer);
#endif

  INSTANTIATE_CLASS(GramLayer);
  REGISTER_LAYER_CLASS(Gram);

}  // namespace caffe
