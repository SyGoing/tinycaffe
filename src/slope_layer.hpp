#ifndef CAFFE_SLOPE_LAYER_HPP_
#define CAFFE_SLOPE_LAYER_HPP_

#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SlopeLayer : public Layer<Dtype> {
 public:
  explicit SlopeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Slope"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);



  Blob<Dtype> concat_blob_;
};

}  // namespace caffe

#endif  // CAFFE_SLOPE_LAYER_HPP_
