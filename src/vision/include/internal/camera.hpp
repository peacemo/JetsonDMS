#ifndef VISION_CAMERA_HPP
#define VISION_CAMERA_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace vision {

/**
 * Camera class
 * Handles camera initialization and frame capturing
 */
class Camera {
public:
    Camera();
    ~Camera();

    /**
     * Initialize camera with device ID
     * @param camera_id Camera device ID
     * @return true on success, false on failure
     */
    bool init(int camera_id);

    /**
     * Check if camera is online and ready
     * @return true if camera is ready
     */
    bool isReady() const;

    /**
     * Capture a single frame from camera
     * @return Captured frame on success, empty cv::Mat on failure
     */
    cv::Mat captureFrame();

    /**
     * Release camera resources
     */
    void release();

private:
    std::string buildCsiPipeline(int camera_id, int width, int height, int fps) const;
    bool openCsiCamera(int camera_id, int width, int height, int fps);

    cv::VideoCapture cap_;
    int camera_id_;
    bool is_initialized_;
    int width_;
    int height_;
    int fps_;
};

} // namespace vision

#endif // VISION_CAMERA_HPP
