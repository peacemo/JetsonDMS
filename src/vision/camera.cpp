#include "include/internal/camera.hpp"
#include <iostream>
#include <sstream>

namespace vision {

namespace {
constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
constexpr int kDefaultFps = 30;
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

    if (!cap_.read(frame) || frame.empty()) {
        std::cerr << "[Camera] Failed to capture frame" << std::endl;
        return cv::Mat();
    }

    return frame;
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
        << " ! appsink";

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

    return true;
}

} // namespace vision
