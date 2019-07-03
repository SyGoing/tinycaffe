#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "blob.hpp"
#include "layer.hpp"
#include "caffe.pb.h"

namespace caffe {

  /**
  * @brief Pools the input image by taking the max, average, etc. within regions.
  *
  * TODO(dox): thorough documentation for Forward, Backward, and proto params.
  */

#define PARTSNUM  9

  template <typename Dtype>
  class PoolingLayer : public Layer<Dtype> {
  public:
    explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Pooling"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int MinTopBlobs() const { return 1; }
    // MAX POOL layers can output an extra top blob for the mask;
    // others can only output the pooled inputs.
    virtual inline int MaxTopBlobs() const {
      return (this->layer_param_.pooling_param().pool() ==
              PoolingParameter_PoolMethod_MAX) ? 2 : 1;
    }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    int channels_;
    int height_, width_;
    int pooled_height_, pooled_width_;
    bool global_pooling_;
    Blob<Dtype> rand_idx_;
    Blob<int> max_idx_;

    // parameters for deformation layer
    virtual void dt(int dims0, int dims1,
                    const Dtype * vals, Dtype ax, Dtype bx, Dtype ay, Dtype by, int n, int ch);

    virtual void reduce1dtran(const Dtype *src, int sheight, Dtype *dst, int dheight,
                              int width, int chan);

    virtual void reduce1dtran(Dtype *src, int sheight, Dtype *dst, int dheight,
                              int width, int chan);

    int N_;
    int Flag_;
    int parts_num_;
    int hpos_[PARTSNUM];
    int vpos_[PARTSNUM];
    double blobl_a_min_;
    /*  vector<Dtype> defw1_;
    vector<Dtype> defw2_;
    vector<Dtype> defw3_;
    vector<Dtype> defw4_;
    vector<Dtype> d_defw1_;
    vector<Dtype> d_defw2_;
    vector<Dtype> d_defw3_;
    vector<Dtype> d_defw4_;*/

    vector<int> defh_;
    vector<int> defv_;
    vector<int> defp_;
    vector<int>Ih_;
    vector<int>Iv_;
    vector<int>tmpIx_;
    vector<int>tmpIy_;
    vector<Dtype>Mdt_;
    vector<Dtype>tmpM_;
    /*
    shared_ptr<SyncedMemory> defh_;
    shared_ptr<SyncedMemory> defv_;
    shared_ptr<SyncedMemory> defp_;
    shared_ptr<SyncedMemory>Ih_;
    shared_ptr<SyncedMemory>Iv_;
    shared_ptr<SyncedMemory>tmpIx_;
    shared_ptr<SyncedMemory>tmpIy_;
    shared_ptr<SyncedMemory>Mdt_;
    shared_ptr<SyncedMemory>tmpM_;
    */

    //  Blob<Dtype> col_buffer_;
    Blob<Dtype> top_buffer_;
    Blob<Dtype> dh_;
    Blob<Dtype> dv_;
    Blob<Dtype> dh2_;
    Blob<Dtype> dv2_;
    int kernel_size_;
    int stride_;
    int pad_;
  };

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
