#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <experimental/filesystem>

#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/kernels/register.h"

#define TENSOR_HEIGHT 257
#define TENSOR_WIDTH 257

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

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels) {
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
            resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
    auto *params = reinterpret_cast<TfLiteResizeBilinearParams *>(
            malloc(sizeof(TfLiteResizeBilinearParams)));
    params->align_corners = false;
    interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                       nullptr);

    interpreter->AllocateTensors();

    // fill input image
    // in[] are integers, cannot do memcpy() directly
    auto input = interpreter->typed_tensor<float>(0);
    for (int i = 0; i < number_of_pixels; i++) {
        input[i] = in[i];
    }

    // fill new_sizes
    interpreter->typed_tensor<int>(1)[0] = wanted_height;
    interpreter->typed_tensor<int>(1)[1] = wanted_width;

    interpreter->Invoke();

    auto output = interpreter->typed_tensor<float>(2);
    auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

    for (int i = 0; i < output_number_of_pixels; i++) {
        out[i] = (uint8_t) output[i];
    }
}

int main(int ac, const char *const *av) {
    po::options_description desc("C++ segmentation example using TFLite");
    desc.add_options()("help,H", "print help message")(
            "model,M", po::value<std::string>(), "specify pretrained TFLite segmentation model path")(
            "input,I", po::value<std::string>(), "specify file to input")(
            "output,O", po::value<std::string>(), "specify file to output");

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

    // Resize input tensors, if desired.
    interpreter->AllocateTensors();

    // TODO: Input/Read image
    array2d<rgb_pixel> img;
    load_image(img, inputPath);
    printf("Image Size: %ldx%ld\n", img.nc(), img.nr());
    image_window raw_input_win(img, "RAW Input Image");

    // TODO: Resize to [1,257,257,3]
//    array2d<rgb_pixel> in_img(TENSOR_HEIGHT,TENSOR_WIDTH);

//    resize_image(img, in_img, interpolate_bilinear());
//
//    printf("Image Size: %ldx%ld\n", img.nc(), img.nr());
//    image_window input_win(in_img, "Input Image");

    // get warp image after transformation
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
    printf("Input Name: %s\n",interpreter->GetInputName(0));
    float* input = interpreter->typed_input_tensor<float>(0);
    resize<float>(input, in.data(),
                  img.nr(), img.nc(), 3,
                  TENSOR_HEIGHT, TENSOR_WIDTH, 3);


    // Fill `input`.
    dlib::array2d<rgb_pixel> test_img(TENSOR_HEIGHT,TENSOR_WIDTH);
    for (int row=0; row<test_img.nr(); row++) {
        for (int col=0; col<test_img.nc(); col++) {
            long index = (row * TENSOR_WIDTH + col) * 3; // + k;
            test_img[row][col].red = input[index];
            test_img[row][col].green = input[index+1];
            test_img[row][col].blue = input[index+2];
        }
    }
    image_window test_win(test_img, "Test Image");

    // TODO: Inference
    interpreter->Invoke();


    // TODO: get output and convert back to image [1, 257, 257, 1] == [ 257,257]
    printf("Output Name: %s\n",interpreter->GetOutputName(0));
    float* output = interpreter->typed_output_tensor<float>(0);

    dlib::array2d<unsigned char> out_img(TENSOR_HEIGHT,TENSOR_WIDTH);
//    for (int i = 0; i < output_number_of_pixels; i++) {
//        out.push_back((uint8_t)output[i]*255);
//    }
    for (int row=0; row<out_img.nr(); row++) {
        for (int col=0; col<out_img.nc(); col++) {
            long index = (row * TENSOR_WIDTH + col); // + k;
//            long index = row * TENSOR_WIDTH + col;
            out_img[col][row] = (unsigned char)output[index];
        }
    }

    // TODO: save/display output
    image_window output_win(out_img, "Output Image");

    std::cout << "The End!" << std::endl;

    raw_input_win.wait_until_closed();
//    input_win.wait_until_closed();
    test_win.wait_until_closed();
    output_win.wait_until_closed();
}
