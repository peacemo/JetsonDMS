#ifndef VISION_PREPROCESSING_HPP
#define VISION_PREPROCESSING_HPP

#include <opencv2/opencv.hpp>

namespace vision {

/**
 * Image preprocessing class
 * Handles image preprocessing operations
 */
class Preprocessor {
public:
    Preprocessor();
    ~Preprocessor();

    /**
     * Initialize preprocessor with target size
     * @param width Target width
     * @param height Target height
     * @return true on success
     */
    bool init(int width, int height);

    /**
     * Process input image
     * @param input Input image
     * @param output Processed image
     * @return true on success, false on failure
     */
    bool process(const cv::Mat& input, cv::Mat& output);

private:
    int target_width_;
    int target_height_;
    bool is_initialized_;

    /**
     * Resize image to target size
     */
    void resize(const cv::Mat& input, cv::Mat& output);

    /**
     * Normalize image
     */
    void normalize(cv::Mat& image);
};

} // namespace vision

#endif // VISION_PREPROCESSING_HPP
