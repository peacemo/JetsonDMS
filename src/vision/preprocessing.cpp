#include "include/internal/preprocessing.hpp"
#include <iostream>

namespace vision {

Preprocessor::Preprocessor()
    : target_width_(0), target_height_(0), is_initialized_(false) {
}

Preprocessor::~Preprocessor() {
}

bool Preprocessor::init(int width, int height) {
    // TODO: Initialize preprocessor parameters
    
    target_width_ = width;
    target_height_ = height;
    is_initialized_ = true;
    
    std::cout << "[Preprocessor] Initialized with size: " 
              << width << "x" << height << std::endl;
    
    return is_initialized_;
}

bool Preprocessor::process(const cv::Mat& input, cv::Mat& output) {
    if (!is_initialized_) {
        std::cerr << "[Preprocessor] Not initialized" << std::endl;
        return false;
    }

    if (input.empty()) {
        std::cerr << "[Preprocessor] Input image is empty" << std::endl;
        return false;
    }

    cv::Mat rgb_input;
    if (input.channels() == 3) {
        cv::cvtColor(input, rgb_input, cv::COLOR_BGR2RGB);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, rgb_input, cv::COLOR_BGRA2RGB);
    } else if (input.channels() == 1) {
        cv::cvtColor(input, rgb_input, cv::COLOR_GRAY2RGB);
    } else {
        std::cerr << "[Preprocessor] Unsupported input channels: "
                  << input.channels() << std::endl;
        return false;
    }

    resize(rgb_input, output);
    if (output.empty()) {
        std::cerr << "[Preprocessor] Resize failed" << std::endl;
        return false;
    }

    normalize(output);
    return !output.empty();
}

void Preprocessor::resize(const cv::Mat& input, cv::Mat& output) {
    (void)target_width_;
    (void)target_height_;
    cv::resize(input, output, cv::Size(224, 224), 0.0, 0.0, cv::INTER_LINEAR);
}

void Preprocessor::normalize(cv::Mat& image) {
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
}

} // namespace vision
