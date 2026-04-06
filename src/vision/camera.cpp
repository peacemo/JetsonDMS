#include "include/internal/camera.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

namespace vision {

namespace {
constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
constexpr int kDefaultFps = 30;
constexpr int kCaptureRetries = 5;
constexpr int kCaptureRetryDelayMs = 30;
constexpr int kCsiWarmupFrames = 3;
} // namespace

Camera::Camera() 
    : camera_id_(-1),
      is_initialized_(false),
      width_(kDefaultWidth),
      height_(kDefaultHeight),
      fps_(kDefaultFps) {
}

Camera::~Camera() {
    release();
}

bool Camera::init(int camera_id) {
    std::cout << "[Camera] Initializing camera " << camera_id << std::endl;

    release();

    camera_id_ = camera_id;
    is_initialized_ = false;

    // Fast prototype strategy: try Jetson CSI pipeline first, then fallback.
    if (openCsiCamera(camera_id_, width_, height_, fps_)) {
        is_initialized_ = true;
        std::cout << "[Camera] CSI camera " << camera_id_ << " is ready" << std::endl;
        return true;
    }

    std::cerr << "[Camera] CSI open failed, fallback to device ID open: " << camera_id_ << std::endl;
    if (!cap_.open(camera_id_)) {
        std::cerr << "[Camera] Failed to open camera device " << camera_id_ << std::endl;
        camera_id_ = -1;
        return false;
    }

    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap_.set(cv::CAP_PROP_FPS, fps_);

    if (!cap_.isOpened()) {
        std::cerr << "[Camera] Camera device opened then became unavailable" << std::endl;
        camera_id_ = -1;
        return false;
    }

    is_initialized_ = true;
    std::cout << "[Camera] Camera " << camera_id_ << " is ready (fallback path)" << std::endl;

    return is_initialized_;
}

bool Camera::isReady() const {
    return is_initialized_ && cap_.isOpened();
}

cv::Mat Camera::captureFrame() {
    cv::Mat frame;

    if (!isReady()) {
        std::cerr << "[Camera] Camera not initialized" << std::endl;
        return frame;
    }

    for (int attempt = 1; attempt <= kCaptureRetries; ++attempt) {
        if (cap_.read(frame) && !frame.empty()) {
            return frame;
        }

        std::cerr << "[Camera] Capture attempt " << attempt
                  << "/" << kCaptureRetries << " failed" << std::endl;

        if (attempt < kCaptureRetries) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(kCaptureRetryDelayMs));
        }
    }

    std::cerr << "[Camera] Failed to capture frame after retries" << std::endl;
    return cv::Mat();
}

void Camera::release() {
    if (cap_.isOpened() || is_initialized_) {
        std::cout << "[Camera] Releasing camera" << std::endl;
    }

    if (cap_.isOpened()) {
        cap_.release();
    }

    camera_id_ = -1;
    is_initialized_ = false;
}

std::string Camera::buildCsiPipeline(int camera_id, int width, int height, int fps) const {
    std::ostringstream pipeline;
    pipeline
        << "nvarguscamerasrc sensor-id=" << camera_id
        << " ! video/x-raw(memory:NVMM), width=" << width
        << ", height=" << height
        << ", framerate=" << fps << "/1"
        << " ! nvvidconv"
        << " ! video/x-raw, format=BGRx"
        << " ! videoconvert"
        << " ! video/x-raw, format=BGR"
        << " ! appsink max-buffers=1 drop=true sync=false";

    return pipeline.str();
}

bool Camera::openCsiCamera(int camera_id, int width, int height, int fps) {
    const std::string pipeline = buildCsiPipeline(camera_id, width, height, fps);
    std::cout << "[Camera] Trying CSI pipeline: " << pipeline << std::endl;

    if (!cap_.open(pipeline, cv::CAP_GSTREAMER)) {
        std::cerr << "[Camera] Failed to open CSI pipeline for sensor-id=" << camera_id << std::endl;
        return false;
    }

    if (!cap_.isOpened()) {
        std::cerr << "[Camera] CSI pipeline opened then became unavailable" << std::endl;
        return false;
    }

    // Warm up CSI stream to avoid unstable first frames.
    cv::Mat warmup_frame;
    for (int i = 0; i < kCsiWarmupFrames; ++i) {
        cap_.read(warmup_frame);
    }

    return true;
}

} // namespace vision
