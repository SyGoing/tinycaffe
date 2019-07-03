#include <vector>

#include "filler.hpp"
#include "layer.hpp"
#include "im2col.hpp"
#include "math_functions.hpp"
#include "custom_layers.hpp"

namespace caffe {



template <typename Dtype>
__global__ void local_update1_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
                                    Dtype* data_R, const int filter_num,
                                    const int location_num, const int output_num) {
  int total = filter_num * location_num * output_num;
  CUDA_KERNEL_LOOP(index, total) {
    int p = index % location_num;
    int n = (index / location_num) % filter_num;
    int q = (index / location_num) / filter_num;
    data_R[index] += data_A[q*location_num+p] * data_B[n*location_num+p];
  }
}

template <typename Dtype>
void local_update1_gpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int filter_num,
                       const int location_num, const int output_num) {
  // data_A is output_num x location_num
  // data_B is filter_num x location_num
  // data_R is output_num x filter_num x location_num, the update performed is Rqnp += Aqp * Bnp

  // NOLINT_NEXT_LINE(whitespace/operators)
  local_update1_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(filter_num * location_num * output_num),
                             CAFFE_CUDA_NUM_THREADS>>>(data_A, data_B, data_R, filter_num, location_num, output_num);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void local_update1_gpu<float>(const float* data_A, const float* data_B,
                                float* data_R, const int filter_num,
                                const int location_num, const int output_num);
template void local_update1_gpu<double>(const double* data_A, const double* data_B,
                                double* data_R, const int filter_num,
                                const int location_num, const int output_num);


template <typename Dtype>
__global__ void local_update2_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
                                Dtype* data_R, const int filter_num,
                                const int location_num, const int output_num) {
  int total = filter_num * location_num;
  CUDA_KERNEL_LOOP(index, total) {
    int p = index % location_num;
    int n = (index / location_num);
    for (int q=0; q<output_num; q++) {
      data_R[index] += data_A[q*location_num+p] * data_B[(q*filter_num+n)*location_num+p];
    }
  }
}

template <typename Dtype>
void local_update2_gpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int filter_num,
                       const int location_num, const int output_num) {
  // data_A is output_num x location_num
  // data_B is output_num x filter_num x location_num
  // data_R is filter_num x location_num, the update performed is Rnp += \sum_q(Aqp * Bqnp)

  // NOLINT_NEXT_LINE(whitespace/operators)
  local_update2_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(filter_num * location_num),
                             CAFFE_CUDA_NUM_THREADS>>>(data_A, data_B, data_R, filter_num, location_num, output_num);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void local_update2_gpu<float>(const float* data_A, const float* data_B,
                       float* data_R, const int filter_num,
                       const int location_num, const int output_num);
template void local_update2_gpu<double>(const double* data_A, const double* data_B,
                       double* data_R, const int filter_num,
                       const int location_num, const int output_num);




/// @brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Dtype* x_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  Blob<Dtype> E;
  E.Reshape(1, 1, 1, K_);
  FillerParameter filler_param;
  filler_param.set_value(1);
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(&E);

  Blob<Dtype> intermediate;
  intermediate.Reshape(1, 1, K_, N_);
  for (int n=0; n<num_; n++) {
    im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
               width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, 1, 1, x_data);

    for (int m=0; m<num_output_; m++) {
      caffe_gpu_mul(K_*N_, x_data, weight+this->blobs_[0]->offset(m),
                    intermediate.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, N_, K_,
                            (Dtype)1., E.gpu_data(), intermediate.gpu_data(),
                            (Dtype)0., top_data + top[0]->offset(n, m));
    }

    if (bias_term_) {
      caffe_gpu_add(M_ * N_, this->blobs_[1]->gpu_data(),
                    top_data + top[0]->offset(n),
                    top_data + top[0]->offset(n));
    }
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(LocalLayer);

}  // namespace caffe