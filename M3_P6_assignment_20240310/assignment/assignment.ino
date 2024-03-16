// Import TensorFlow stuff
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
//#include <tensorflow/lite/version.h>

// Our model
#include "model_tflite_quant.h"

#include "test_data.h"

// Figure out what's going on in our model
#define DEBUG 1


// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow
// arrays. You'll need to adjust this by combiling, running, and looking
// for errors.
constexpr int kTensorArenaSize = 5 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t* model_input_buffer = nullptr;

} // namespace


void setup() {
  // Wait for Serial to connect
#if DEBUG
  Serial.begin(115200);
  while (!Serial);
#endif


  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(model_tflite_quant);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while (1);
  }

  tflite::AllOpsResolver tflOpsResolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, tflOpsResolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while (1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_input_buffer = model_input->data.int8;
  model_output = interpreter->output(0);

  // Get information about the memory area to use for the model's input
  // Supported data types:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif


}

void loop() {
  int y_val[10] = {0};
  int invoke_status = 0;

  memcpy(model_input_buffer, test_data, 784); //Copy the data from test_data.h to check the network
  invoke_status = 1;

  if (invoke_status == 1)
  {
    invoke_status = 0;

    //Invoking the tflite interpreter
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return;
    }

    // Read predicted y value from output buffer (tensor)
//    while (1)
//    {
      for (int i = 0; i < 10; i++)
      {
        y_val[i] = model_output->data.int8[i];
        Serial.print(i);
        Serial.print("  ");
        Serial.println(y_val[i]);
      }
      Serial.println("........................");
//    }



  }
}
