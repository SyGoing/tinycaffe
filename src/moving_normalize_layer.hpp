#ifndef CAFFE_NORMALIZE_LAYER_HPP_
#define CAFFE_NORMALIZE_LAYER_HPP_

#include <utility>
#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "caffe.pb.h"

namespace caffe {
  /**
  * @brief MovingNormalizes input.
  */
  template <typename Dtype>
  class MovingNormalizeLayer : public Layer<Dtype> {
  public:
    explicit MovingNormalizeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "MovingNormalize"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);


    Blob<Dtype> sum_multiplier_, squared_, norm_;
    int iter_;
  };

}

#endif  // CAFFE_NORMALIZE_LAYER_HPP_
