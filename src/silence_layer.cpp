#include <vector>

#include "silence_layer.hpp"
#include "math_functions.hpp"

namespace caffe {



#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS(SilenceLayer);
REGISTER_LAYER_CLASS(Silence);

}  // namespace caffe
