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
     * @param frame Output frame data
     * @return true on success, false on failure
     */
    bool captureFrame(cv::Mat& frame);

    /**
     * Release camera resources
     */
    void release();

private:
    cv::VideoCapture cap_;
    int camera_id_;
    bool is_initialized_;
};

} // namespace vision

#endif // VISION_CAMERA_HPP
