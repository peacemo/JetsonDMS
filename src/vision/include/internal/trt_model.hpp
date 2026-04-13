#ifndef VISION_TRT_MODEL_HPP
#define VISION_TRT_MODEL_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <string>
#include <vector>
#include "../vision_types.h"

namespace vision {

/**
 * TensorRT inference class
 * Handles model loading and inference
 */
class TrtModel {
public:
    TrtModel();
    ~TrtModel();

    /**
     * Initialize inference engine with model
     * @param model_path Path to TensorRT model file
     * @return true on success, false on failure
     */
    bool init(const std::string& model_path);

    /**
     * Perform inference on preprocessed image
     * @param image Input preprocessed image
     * @param result Output detection result
     * @return true on success, false on failure
     */
    bool infer(const cv::Mat& image, DetectResult& result);

    /**
     * Release inference resources
     */
    void release();

private:
    std::string model_path_;
    bool is_initialized_;
    int frame_counter_;

    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    cudaStream_t stream_;

    std::string input_tensor_name_;
    std::string output_tensor_name_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    nvinfer1::DataType input_data_type_;
    nvinfer1::DataType output_data_type_;

    bool input_is_dynamic_;
    std::size_t input_elements_;
    std::size_t output_elements_;
    std::size_t input_bytes_;
    std::size_t output_bytes_;

    void* input_device_buffer_;
    void* output_device_buffer_;

    std::vector<float> input_host_f32_;
    std::vector<std::uint16_t> input_host_f16_;
    std::vector<float> output_host_f32_;
    std::vector<std::uint16_t> output_host_f16_;

    /**
     * Convert OpenCV Mat to TensorRT input format
     */
    bool convertToTensorRT(const cv::Mat& image);

    /**
     * Process model output to detection result
     */
    bool processOutput(DetectResult& result);
};

} // namespace vision

#endif // VISION_TRT_MODEL_HPP
