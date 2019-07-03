#include <vector>

#include "filler.hpp"
#include "layer.hpp"
#include "im2col.hpp"
#include "math_functions.hpp"
#include "resize_layer.hpp"
#include "util_img.hpp"
namespace caffe {


  template <typename Dtype>
  void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    ResizeBlob_gpu(bottom[0], top[0]);
  }


  INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);

}  // namespace caffe