#include <stdio.h>
#include "vision/include/vision_api.h"

int main() {
    printf("=== Hello World ===\n");
    
    // Initialize vision system
    printf("Initializing vision system...\n");
    int ret = vision_init(0, "models/dms_model.engine");
    if (ret != 0) {
        printf("Failed to initialize vision system\n");
        return 1;
    }
    
    // Perform detection
    DetectResult result;
    printf("Running detection...\n");
    ret = vision_detect(&result);
    if (ret != 0) {
        printf("Detection failed\n");
    } else {
        printf("Detection result: status=%d, confidence=%.2f, frame_id=%d\n",
               result.status, result.confidence, result.frame_id);
    }
    
    // Cleanup
    printf("Cleaning up...\n");
    vision_cleanup();
    
    printf("=== Test Complete ===\n");
    return 0;
}
