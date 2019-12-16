#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

#ifdef MODEL_DIR
#define MODEL_DIR_ MODEL_DIR
#else
#define MODEL_DIR_ "../models/deeplabv3_257_mv_gpu.tflite"
#endif

// See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/minimal.cc
// See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/label_image/label_image.cc

int main(int ac, char* av[])
{
    std::string graph_path = std::string(MODEL_DIR_)+"/deeplabv3_257_mv_gpu.tflite";
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

//    float* input = interpreter->typed_input_tensor<float>(0);
//    // Fill `input`.
//
//    interpreter->Invoke();
//
//    float* output = interpreter->typed_output_tensor<float>(0);

    std::cout << "Hello World!" << std::endl;
}
