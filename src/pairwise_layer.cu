#include <cfloat>
#include <vector>

#include "pairwise_layer.hpp"
#include "math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  __global__ void ProdForward(const int M_, const int N_, const int K_,
                               const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                               Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, M_ * N_ * K_) {
      int m = index / N_ / K_;
      int nk = index % (N_* K_);
      int n = nk / K_;
      int k = nk % K_;
      top_data[index] = bottom_data_a[m * K_ + k] * bottom_data_b[n * K_ + k];
    }
  }

  template <typename Dtype>
  __global__ void SumForward(const int M_, const int N_, const int K_,
                              const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                              Dtype coeff0, Dtype coeff1,
                              Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, M_ * N_ * K_) {
      int m = index / N_ / K_;
      int nk = index % (N_* K_);
      int n = nk / K_;
      int k = nk % K_;
      top_data[index] = bottom_data_a[m * K_ + k] *coeff0 + bottom_data_b[n * K_ + k] * coeff1;
    }
  }

  template <typename Dtype>
  __global__ void MaxForward(const int M_, const int N_, const int K_,
                             const Dtype* bottom_data_a, const Dtype* bottom_data_b,
                             Dtype* top_data, int* mask) {
    CUDA_KERNEL_LOOP(index, M_ * N_ * K_) {
      int m = index / N_ / K_;
      int nk = index % (N_* K_);
      int n = nk / K_;
      int k = nk % K_;
      if (bottom_data_a[m * K_ + k] > bottom_data_b[n * K_ + k]) {
        top_data[index] = bottom_data_a[m * K_ + k];
        mask[index] = 0;
      }
      else {
        top_data[index] = bottom_data_b[n * K_ + k];
        mask[index] = 1;
      }
    }
  }

template <typename Dtype>
void PairwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_gpu_data();
  switch (op_) {
  case PairwiseParameter_PairwiseOp_PROD:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ProdForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
    break;
  case PairwiseParameter_PairwiseOp_SUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SumForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), coeffs_[0], coeffs_[1], top_data);
    break;
  case PairwiseParameter_PairwiseOp_MAX:
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      M_, N_, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data, mask);
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}



INSTANTIATE_LAYER_GPU_FUNCS(PairwiseLayer);

}  // namespace caffe
