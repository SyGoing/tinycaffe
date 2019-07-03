#ifndef CAFFE_NOISE_LAYER_HPP_
#define CAFFE_NOISE_LAYER_HPP_

#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "caffe.pb.h"

#include "neuron_layer.hpp"

namespace caffe {
  /**
  * @brief Add noise.
  */
  template <typename Dtype>
  class NoiseLayer : public NeuronLayer<Dtype> {
  public:
    explicit NoiseLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Noise"; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);


    Blob<Dtype> mask;
  };

}  // namespace caffe

#endif  // CAFFE_NOISE_LAYER_HPP_