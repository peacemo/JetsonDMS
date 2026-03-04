#ifndef VISION_TRT_MODEL_HPP
#define VISION_TRT_MODEL_HPP

#include <opencv2/opencv.hpp>
#include <string>
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

    // TensorRT related members (to be implemented)
    // void* engine_;
    // void* context_;

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
