#include "include/internal/vision_impl.hpp"
#include <iostream>

namespace vision {

VisionPipeline::VisionPipeline()
    : is_initialized_(false) {
    // Create component instances
    camera_ = std::make_unique<Camera>();
    preprocessor_ = std::make_unique<Preprocessor>();
    inference_ = std::make_unique<TrtModel>();
}

VisionPipeline::~VisionPipeline() {
    cleanup();
}

bool VisionPipeline::init(int camera_id, const std::string& model_path) {
    // TODO: Initialize all components in order
    
    std::cout << "[VisionPipeline] Initializing..." << std::endl;
    
    // Initialize camera
    if (!camera_->init(camera_id)) {
        std::cerr << "[VisionPipeline] Failed to initialize camera" << std::endl;
        return false;
    }
    
    // Initialize preprocessor (typical model input size: 224x224 or 640x640)
    if (!preprocessor_->init(224, 224)) {
        std::cerr << "[VisionPipeline] Failed to initialize preprocessor" << std::endl;
        return false;
    }
    
    // Initialize inference
    if (!inference_->init(model_path)) {
        std::cerr << "[VisionPipeline] Failed to initialize inference" << std::endl;
        return false;
    }
    
    is_initialized_ = true;
    std::cout << "[VisionPipeline] Initialization complete" << std::endl;
    
    return is_initialized_;
}

bool VisionPipeline::detect(DetectResult& result) {
    // TODO: Execute full detection pipeline
    
    if (!is_initialized_) {
        std::cerr << "[VisionPipeline] Pipeline not initialized" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        return false;
    }
    
    // Step 1: Capture frame
    raw_frame_ = camera_->captureFrame();
    if (raw_frame_.empty()) {
        std::cerr << "[VisionPipeline] Failed to capture frame" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        return false;
    }
    
    // Step 2: Preprocess image
    if (!preprocessor_->process(raw_frame_, processed_frame_)) {
        std::cerr << "[VisionPipeline] Failed to preprocess image" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        return false;
    }
    
    // Step 3: Run inference
    if (!inference_->infer(processed_frame_, result)) {
        std::cerr << "[VisionPipeline] Failed to run inference" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        return false;
    }
    
    return true;
}

void VisionPipeline::cleanup() {
    // TODO: Cleanup all components
    
    if (is_initialized_) {
        std::cout << "[VisionPipeline] Cleaning up..." << std::endl;
        
        if (inference_) inference_->release();
        if (camera_) camera_->release();
        
        is_initialized_ = false;
    }
}

bool VisionPipeline::isReady() const {
    return is_initialized_ && 
           camera_->isReady();
}

} // namespace vision
