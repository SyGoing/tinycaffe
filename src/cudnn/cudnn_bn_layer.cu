#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "../layer.hpp"
#include "../math_functions.hpp"
#include "../filler.hpp"
#include "cudnn_bn_layer.hpp"

#if CUDNN_VERSION_MIN(4, 0, 0)

namespace caffe {

  template <typename Dtype>
  void CuDNNBNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* scale_data = this->blobs_[0]->gpu_data();
    const Dtype* bias_data = this->blobs_[1]->gpu_data();

    if (this->phase_ == TEST) {
      const Dtype* running_mean_data = this->blobs_[2]->gpu_data();
      const Dtype* running_inv_variance_data = this->blobs_[3]->gpu_data();
      CUDNN_CHECK(cudnnBatchNormalizationForwardInference(handle_,
                                                          CUDNN_BATCHNORM_SPATIAL,
                                                          cudnn::dataType<Dtype>::one,
                                                          cudnn::dataType<Dtype>::zero,
                                                          bottom_desc_, bottom_data,
                                                          top_desc_, top_data,
                                                          bn_param_desc_, scale_data, bias_data,
                                                          running_mean_data, running_inv_variance_data,
                                                          this->bn_eps_));
    }
    else {
      Dtype* running_mean_data = this->blobs_[2]->mutable_gpu_data();
      Dtype* running_inv_variance_data = this->blobs_[3]->mutable_gpu_data();
      Dtype* save_mean_data = save_mean_.mutable_gpu_data();
      Dtype* save_inv_variance_data = save_inv_variance_.mutable_gpu_data();
      CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(handle_,
                                                         CUDNN_BATCHNORM_SPATIAL,
                                                         cudnn::dataType<Dtype>::one,
                                                         cudnn::dataType<Dtype>::zero,
                                                         bottom_desc_, bottom_data,
                                                         top_desc_, top_data,
                                                         bn_param_desc_, scale_data, bias_data,
                                                         this->bn_momentum_,
                                                         running_mean_data, running_inv_variance_data,
                                                         this->bn_eps_,
                                                         save_mean_data, save_inv_variance_data));
    }
  }


  INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBNLayer);

}  // namespace caffe
#endif
#endif
