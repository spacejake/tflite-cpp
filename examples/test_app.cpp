#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/uio.h>
#include <string>
#include <vector>
#include <random>

#include <functional>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <climits>


#include <boost/program_options.hpp>
#include <experimental/filesystem>

#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/evaluation/utils.h"


#define LOG(x) std::cerr

#define TENSOR_HEIGHT 257
#define TENSOR_WIDTH 257

#define IMAGE_MEAN 128.0f
#define IMAGE_STD 128.0f
#define NUM_CLASSES 21
#define COLOR_CHANNELS 3

// See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc
// See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc
using namespace dlib;
namespace {
    namespace fs = std::experimental::filesystem;
    namespace po = boost::program_options;
} // namespace


std::string parseStringFromOpts(const po::variables_map& vm, const std::string& id) {
    if (vm.count(id) == 0) {
        return std::string();
    }

    return vm[id].as<std::string>();
}

std::string parsePathFromOpts(const po::variables_map& vm, const std::string& id) {
    std::string path = parseStringFromOpts(vm, id);

    printf("Opts Passed: %s\n", path.c_str());
    if (!fs::exists(path)) {
        return std::string();
    }

    return path;
}

using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

TfLiteDelegatePtr CreateGPUDelegate(tflite::FlatBufferModel *model) {
#if defined(__ANDROID__)
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
  gpu_opts.inference_preference =
      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  gpu_opts.inference_priority1 =
      s->allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                    : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  return tflite::evaluation::CreateGPUDelegate(model, &gpu_opts);
#else
    return tflite::evaluation::CreateGPUDelegate(model);
#endif
}

TfLiteDelegatePtrMap GetDelegates(tflite::FlatBufferModel* model) {
    TfLiteDelegatePtrMap delegates;
    auto delegate = CreateGPUDelegate(model);
    if (!delegate) {
        LOG(INFO) << "GPU acceleration is unsupported on this platform.";
    } else {
        delegates.emplace("GPU", std::move(delegate));
    }

    return delegates;
}

rgb_alpha_pixel rand_rgba(uint8_t alpha=255) {
    std::random_device r;

    auto rand_uchar = std::bind(std::uniform_int_distribution<>(0, UCHAR_MAX),
                          std::mt19937(r()));

    return rgb_alpha_pixel(
            rand_uchar(),
            rand_uchar(),
            rand_uchar(),
            alpha);
}


template <class OUT_T, class IN_T>
void resize(OUT_T* out, IN_T* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, bool normalize=true,
            tflite::BuiltinOperator op=tflite::BuiltinOperator_RESIZE_BILINEAR) {
    int number_of_pixels = image_height * image_width * image_channels;
    std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

    int base_index = 0;

    // two inputs: input and new_sizes
    interpreter->AddTensors(2, &base_index);
    // one output
    interpreter->AddTensors(1, &base_index);
    // set input and output tensors
    interpreter->SetInputs({0, 1});
    interpreter->SetOutputs({2});

    // set parameters of tensors
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(
            0, kTfLiteFloat32, "input",
            {1, image_height, image_width, image_channels}, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                              quant);
    interpreter->SetTensorParametersReadWrite(
            2, kTfLiteFloat32, "output",
            {1, wanted_height, wanted_width, wanted_channels}, quant);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    const TfLiteRegistration *resize_op =
            resolver.FindOp(op, 1);
    auto *params = reinterpret_cast<TfLiteResizeBilinearParams *>(
            malloc(sizeof(TfLiteResizeBilinearParams)));
    params->align_corners = false;
    interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                       nullptr);

    interpreter->AllocateTensors();

    // fill input image
    // in[] are integers, cannot do memcpy() directly
    auto input = interpreter->typed_tensor<OUT_T>(0);
    for (int i = 0; i < number_of_pixels; i++) {
        input[i] = in[i];
    }

    // fill new_sizes
    interpreter->typed_tensor<int>(1)[0] = wanted_height;
    interpreter->typed_tensor<int>(1)[1] = wanted_width;

    interpreter->Invoke();

    auto output = interpreter->typed_tensor<OUT_T>(2);
    auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

    for (int i = 0; i < output_number_of_pixels; i++) {
        if (normalize) {
            out[i] = ((OUT_T) output[i] - IMAGE_MEAN) / IMAGE_STD;
        } else {
            out[i] = (OUT_T) output[i];
        }
    }
}

int main(int ac, const char *const *av) {
    bool use_gpu = false;

    po::options_description desc("C++ segmentation example using TFLite");
    desc.add_options()("help,H", "print help message")(
            "model,M", po::value<std::string>(), "specify pretrained TFLite segmentation model path")(
            "input,I", po::value<std::string>(), "specify file to input")(
            "output,O", po::value<std::string>(), "specify file to output")(
            "gpu,G", po::bool_switch(&use_gpu), "description");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 1;
    }

    std::string graph_path = parsePathFromOpts(vm, "model");
    if (graph_path.size() == 0) {
        std::cout << "Please specify valid 'detector path'" << std::endl;
        return 1;
    }

    std::string inputPath = parsePathFromOpts(vm, "input");
    if (inputPath.size() == 0) {
        std::cout << "Please specify valid 'input path'" << std::endl;
        return 1;
    }

    std::string outputPath = parseStringFromOpts(vm, "output");
    if (outputPath.size() == 0) {
        std::cout << "Please specify 'output path'" << std::endl;
        return 1;
    }

    const int num_threads = 1;
    std::string input_layer_type = "float";
    std::vector<int> sizes = {2};
    float x,y;

    std::unique_ptr<tflite::FlatBufferModel> model(
            tflite::FlatBufferModel::BuildFromFile(graph_path.c_str()));

    if(!model){
        printf("Failed to model\n");
        exit(0);
    } else {
        printf("Loaded model: %s\n", graph_path.c_str());
    }
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    auto tensor_size = interpreter->tensors_size();
    std::cout << "GRAPH Size: " << tensor_size;
    for (int idx = 0; idx < tensor_size; ++idx) {
        std::cout << "\n\tTensor Name[" << idx << "]: " << interpreter->tensor(idx)->name << " Size: ";
        for (int i = 0; i < interpreter->tensor(idx)->dims->size; ++i) {
            std::cout << interpreter->tensor(idx)->dims->data[i] << " ";
        }
        std::cout << std::endl;
    }

    if (use_gpu) {
        auto delegates_ = GetDelegates(model.get());
        for (const auto &delegate : delegates_) {
            if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) !=
                kTfLiteOk) {
                LOG(FATAL) << "Failed to apply " << delegate.first << " delegate.";
            } else {
                LOG(INFO) << "Applied " << delegate.first << " delegate.";
            }
        }
    }


    // Resize input tensors, if desired.
    interpreter->AllocateTensors();

    // TODO: Input/Read image
    array2d<rgb_pixel> img;
    load_image(img, inputPath);
    printf("Image Size: %ldx%ld\n", img.nc(), img.nr());
    image_window raw_input_win(img, "RAW Input Image");

    // TODO: Resize to [1,257,257,3]
    std::vector<uint8_t> in;

    // iterate over rows & cols
    for (int row=0; row<img.nr(); row++) {
        for (int col=0; col<img.nc(); col++) {
            in.push_back(img[row][col].red);
            in.push_back(img[row][col].green);
            in.push_back(img[row][col].blue);
        }
    }

    // TODO: Convert to tensor
//    printf("Input Name: %s, %d\n",interpreter->GetInputName(0), interpreter->tensor(0)->dims[0]);
    int input_idx = interpreter->inputs()[0];
    std::cout << "Input Name: " << interpreter->GetInputName(0) << "/" << interpreter->tensor(input_idx)->name << " Size: ";
    for (int i = 0; i < interpreter->tensor(input_idx)->dims->size; ++i) {
        std::cout << interpreter->tensor(input_idx)->dims->data[i] << " ";
    }
    std::cout << std::endl;

    float* input = interpreter->typed_input_tensor<float>(0);
    resize<float>(input, in.data(),
                  img.nr(), img.nc(), COLOR_CHANNELS,
                  TENSOR_HEIGHT, TENSOR_WIDTH, COLOR_CHANNELS);


    // Fill `input`.
    dlib::array2d<rgb_pixel> test_img(TENSOR_HEIGHT,TENSOR_WIDTH);
    for (int row=0; row<test_img.nr(); row++) {
        for (int col=0; col<test_img.nc(); col++) {
            long index = (row * TENSOR_WIDTH + col) * COLOR_CHANNELS; // + k;
            test_img[row][col].red = input[index]*IMAGE_STD+IMAGE_MEAN;
            test_img[row][col].green = input[index+1]*IMAGE_STD+IMAGE_MEAN;
            test_img[row][col].blue = input[index+2]*IMAGE_STD+IMAGE_MEAN;
        }
    }
    image_window test_win(test_img, "Test Image");

    // TODO: Inference
    interpreter->Invoke();


    // TODO: get output and convert back to image [1, 257, 257, 1] == [ 257,257]
//    printf("Output Name: %s\n",interpreter->GetOutputName(0));
    float* output = interpreter->typed_output_tensor<float>(0);
    //    int output_idx = interpreter->outputs()[0];
    for (int idx = 0; idx < interpreter->outputs().size(); ++idx) {
        auto output_idx = interpreter->outputs()[idx];
        std::cout << "Output Name[" << idx << "->" << output_idx << "]: " << interpreter->GetOutputName(idx) << "/"
                  << interpreter->tensor(output_idx)->name << " Size: ";
        for (int i = 0; i < interpreter->tensor(output_idx)->dims->size; ++i) {
            std::cout << interpreter->tensor(output_idx)->dims->data[i] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> result(img.nr() * img.nc() * NUM_CLASSES);
    resize<float>(result.data(), output,
                  TENSOR_HEIGHT, TENSOR_WIDTH, NUM_CLASSES,
                  img.nr(), img.nc(), NUM_CLASSES,
                  true);

    dlib::array2d<rgb_alpha_pixel> out_img(img.nr(), img.nc());
    std::vector<rgb_alpha_pixel> mSegmentColors;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (i == 0) {
            mSegmentColors.push_back(rgb_alpha_pixel(0,0,0,0)); // Transparent
        } else {
            mSegmentColors.push_back(rand_rgba(150));
        }
    }

    float maxVal = 0;
    for (int row=0; row<img.nr(); row++) {
        for (int col=0; col<img.nc(); col++) {
            int mSegmentBits = 0;

            for (int c = 0; c < NUM_CLASSES; c++) {
                long index = (row * img.nc() * NUM_CLASSES + col * NUM_CLASSES + c);
                float val = result[index];
                if (c == 0 || val > maxVal) {
                    maxVal = val;
                    mSegmentBits = c;
                }
            }

            out_img[row][col] = mSegmentColors[mSegmentBits];
        }
    }

    // TODO: save/display output
    image_window output_win(out_img, "Output Image");

    save_png(out_img, outputPath);

    std::cout << "The End!" << std::endl;

    raw_input_win.wait_until_closed();
//    input_win.wait_until_closed();
    test_win.wait_until_closed();
    output_win.wait_until_closed();
}
