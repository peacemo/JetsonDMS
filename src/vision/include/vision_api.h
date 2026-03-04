#ifndef VISION_API_H
#define VISION_API_H

#include "vision_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the vision detection system
 * @param camera_id Camera device ID (default: 0)
 * @param model_path Path to TensorRT model file
 * @return 0 on success, -1 on failure
 */
int vision_init(int camera_id, const char* model_path);

/**
 * Perform detection on current camera frame
 * @param result Pointer to store detection result
 * @return 0 on success, -1 on failure
 */
int vision_detect(DetectResult* result);

/**
 * Clean up and release vision system resources
 */
void vision_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // VISION_API_H
