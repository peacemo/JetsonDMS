#include "include/vision_api.h"
#include "include/internal/vision_impl.hpp"
#include <iostream>
#include <memory>

// Global vision pipeline instance
static std::unique_ptr<vision::VisionPipeline> g_pipeline = nullptr;

extern "C" {

int vision_init(int camera_id, const char* model_path) {
    // TODO: Implement C API initialization wrapper
    
    try {
        std::cout << "[C API] Initializing vision system..." << std::endl;
        
        // Check if already initialized
        if (g_pipeline != nullptr) {
            std::cerr << "[C API] Vision system already initialized" << std::endl;
            return -1;
        }
        
        // Validate input parameters
        if (model_path == nullptr) {
            std::cerr << "[C API] Invalid model path" << std::endl;
            return -1;
        }
        
        // Create pipeline instance
        g_pipeline = std::make_unique<vision::VisionPipeline>();
        
        // Initialize pipeline
        if (!g_pipeline->init(camera_id, std::string(model_path))) {
            std::cerr << "[C API] Failed to initialize pipeline" << std::endl;
            g_pipeline.reset();
            return -1;
        }
        
        std::cout << "[C API] Vision system initialized successfully" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[C API] Exception during init: " << e.what() << std::endl;
        g_pipeline.reset();
        return -1;
    }
}

int vision_detect(DetectResult* result) {
    // TODO: Implement C API detection wrapper
    
    try {
        // Validate pipeline exists
        if (g_pipeline == nullptr) {
            std::cerr << "[C API] Vision system not initialized" << std::endl;
            if (result) {
                result->status = DETECT_STATUS_ERROR;
                result->confidence = 0.0f;
                result->frame_id = -1;
            }
            return -1;
        }
        
        // Validate result pointer
        if (result == nullptr) {
            std::cerr << "[C API] Invalid result pointer" << std::endl;
            return -1;
        }
        
        // Call C++ detection
        if (!g_pipeline->detect(*result)) {
            std::cerr << "[C API] Detection failed" << std::endl;
            return -1;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[C API] Exception during detect: " << e.what() << std::endl;
        if (result) {
            result->status = DETECT_STATUS_ERROR;
        }
        return -1;
    }
}

void vision_cleanup(void) {
    // TODO: Implement C API cleanup wrapper
    
    try {
        std::cout << "[C API] Cleaning up vision system..." << std::endl;
        
        if (g_pipeline != nullptr) {
            g_pipeline->cleanup();
            g_pipeline.reset();
        }
        
        std::cout << "[C API] Vision system cleaned up" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[C API] Exception during cleanup: " << e.what() << std::endl;
    }
}

} // extern "C"
