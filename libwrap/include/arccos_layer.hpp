#ifndef CAFFE_ARCCOS_LAYER_HPP_
#define CAFFE_ARCCOS_LAYER_HPP_

#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "caffe.pb.h"

#include "neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class ArccosLayer : public NeuronLayer<Dtype> {
 public:

  explicit ArccosLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Arccos"; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_ARCCOS_LAYER_HPP_
