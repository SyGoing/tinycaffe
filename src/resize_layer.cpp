#include <vector>

#include "filler.hpp"
#include "layer.hpp"
#include "im2col.hpp"
#include "math_functions.hpp"
#include "resize_layer.hpp"
#include "util_img.hpp"
namespace caffe {

  template <typename Dtype>
  void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
    // Configure the kernel size, padding, stride, and inputs.
    ResizeParameter resize_param = this->layer_param_.resize_param();


    bool is_pyramid_test = resize_param.is_pyramid_test();
    if (is_pyramid_test == false) {
      CHECK(resize_param.has_height()) << "output height is required ";
      CHECK(resize_param.has_width()) << "output width is required ";
      this->out_height_ = resize_param.height();
      this->out_width_ = resize_param.width();
    }
    else {
      CHECK(resize_param.has_out_height_scale()) << "output height scale is required ";
      CHECK(resize_param.has_out_width_scale()) << "output width scale is required ";
      int in_height = bottom[0]->height();
      int in_width = bottom[0]->width();
      this->out_height_ = int(resize_param.out_height_scale() * in_height);
      this->out_width_ = int(resize_param.out_width_scale() * in_width);
    }




    for (int i = 0; i<4; i++) {
      this->locs_.push_back(new Blob<Dtype>);
    }
  }

  template <typename Dtype>
  void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {

    ResizeParameter resize_param = this->layer_param_.resize_param();

    bool is_pyramid_test = resize_param.is_pyramid_test();
    if (is_pyramid_test == false) {

      this->out_height_ = resize_param.height();
      this->out_width_ = resize_param.width();
    }
    else {
      int in_height = bottom[0]->height();
      int in_width = bottom[0]->width();
      this->out_height_ = int(resize_param.out_height_scale() * in_height);
      this->out_width_ = int(resize_param.out_width_scale() * in_width);
    }


    this->out_num_ = bottom[0]->num();
    this->out_channels_ = bottom[0]->channels();
    top[0]->Reshape(out_num_, out_channels_, out_height_, out_width_);

    for (int i = 0; i<4; ++i) {
      this->locs_[i]->Reshape(1, 1, out_height_, out_width_);
    }
  }

  template <typename Dtype>
  void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    //	ResizeBlob_cpu(bottom[0],top[0],this->locs_[0],this->locs_[1],this->locs_[2],this->locs_[3]);
    ResizeBlob_cpu(bottom[0], top[0]);

  }


#ifdef CPU_ONLY
  STUB_GPU(ResizeLayer);
#endif

  INSTANTIATE_CLASS(ResizeLayer);
  REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe