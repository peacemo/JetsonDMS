#ifndef VISION_IMPL_HPP
#define VISION_IMPL_HPP

#include "camera.hpp"
#include "preprocessing.hpp"
#include "trt_model.hpp"
#include "../vision_types.h"
#include <memory>

namespace vision {

/**
 * Vision pipeline implementation
 * Integrates capture, preprocessing, and inference
 */
class VisionPipeline {
public:
    VisionPipeline();
    ~VisionPipeline();

    /**
     * Initialize the complete vision pipeline
     * @param camera_id Camera device ID
     * @param model_path Path to TensorRT model
     * @return true on success, false on failure
     */
    bool init(int camera_id, const std::string& model_path);

    /**
     * Execute full detection pipeline
     * @param result Output detection result
     * @return true on success, false on failure
     */
    bool detect(DetectResult& result);

    /**
     * Clean up all resources
     */
    void cleanup();

    /**
     * Check if pipeline is ready
     */
    bool isReady() const;

private:
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<TrtModel> inference_;
    bool is_initialized_;

    // Internal buffers
    cv::Mat raw_frame_;
    cv::Mat processed_frame_;
};

} // namespace vision

#endif // VISION_IMPL_HPP
