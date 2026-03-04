#ifndef VISION_TYPES_H
#define VISION_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Detection status enumeration
 * Used for both C and C++ code
 */
typedef enum {
    DETECT_STATUS_UNKNOWN = 0,
    DETECT_STATUS_NORMAL = 1,
    DETECT_STATUS_DROWSY = 2,
    DETECT_STATUS_DISTRACTED = 3,
    DETECT_STATUS_ERROR = -1
} DetectStatus;

/**
 * Detection result structure
 * Contains detection status and confidence score
 */
typedef struct {
    DetectStatus status;
    float confidence;
    int frame_id;
} DetectResult;

#ifdef __cplusplus
}
#endif

#endif // VISION_TYPES_H
