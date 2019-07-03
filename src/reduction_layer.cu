#include <vector>

#include "reduction_layer.hpp"
#include "math_functions.hpp"

namespace caffe {

#define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))

  template <typename Dtype>
  __global__ void SUMForward(const int num, const int dim,
                              const Dtype* bottom_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, num) {
      top_data[index] = Dtype(0);
      for (int d = 0; d < dim; d++) {
        top_data[index] += bottom_data[index * dim + d];
      }
    }
  }

  template <typename Dtype>
  __global__ void MeanForward(const int num, const int dim,
                             const Dtype* bottom_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, num) {
      top_data[index] = Dtype(0);
      for (int d = 0; d < dim; d++) {
        top_data[index] += bottom_data[index * dim + d];
      }
      top_data[index] /= dim;
    }
  }

  template <typename Dtype>
  __global__ void SUMSQForward(const int num, const int dim,
                             const Dtype* bottom_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, num) {
      top_data[index] = Dtype(0);
      for (int d = 0; d < dim; d++) {
        top_data[index] += bottom_data[index * dim + d] * bottom_data[index * dim + d];
      }
    }
  }

  template <typename Dtype>
  __global__ void ASUMForward(const int num, const int dim,
                               const Dtype* bottom_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, num) {
      top_data[index] = Dtype(0);
      for (int d = 0; d < dim; d++) {
        top_data[index] += sign(bottom_data[index * dim + d]);
      }
    }
  }

template <typename Dtype>
void ReductionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case ReductionParameter_ReductionOp_SUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SUMForward<Dtype> << <CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS >> >(
      num_, dim_, bottom_data, top_data);
  case ReductionParameter_ReductionOp_MEAN:
    // NOLINT_NEXT_LINE(whitespace/operators)
    MeanForward<Dtype> << <CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS >> >(
      num_, dim_, bottom_data, top_data);
    break;
  case ReductionParameter_ReductionOp_ASUM:
    // NOLINT_NEXT_LINE(whitespace/operators)
    ASUMForward<Dtype> << <CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS >> >(
      num_, dim_, bottom_data, top_data);
    break;
  case ReductionParameter_ReductionOp_SUMSQ:
    // NOLINT_NEXT_LINE(whitespace/operators)
    SUMSQForward<Dtype> << <CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS >> >(
      num_, dim_, bottom_data, top_data);
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
      << ReductionParameter_ReductionOp_Name(op_);
  }
  if (coeff_ != Dtype(1)) {
    // Reset the top_data pointer.
    top_data = top[0]->mutable_gpu_data();
    caffe_gpu_scal(num_, coeff_, top_data);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReductionLayer);

}  // namespace caffe
