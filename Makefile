# Compilers
CC = gcc
CXX = g++

# Directories
SRC_DIR = src
VISION_DIR = $(SRC_DIR)/vision
INCLUDE_DIR = $(VISION_DIR)/include
BUILD_DIR = build

# Target
TARGET = dms

# Sources
C_SRCS = $(SRC_DIR)/main.c
CXX_SRCS = $(VISION_DIR)/camera.cpp \
           $(VISION_DIR)/preprocessing.cpp \
           $(VISION_DIR)/trt_model.cpp \
           $(VISION_DIR)/vision_impl.cpp \
           $(VISION_DIR)/vision_api.cpp

# Objects
C_OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
CXX_OBJS = $(patsubst $(VISION_DIR)/%.cpp,$(BUILD_DIR)/vision/%.o,$(CXX_SRCS))
OBJS = $(C_OBJS) $(CXX_OBJS)

# Flags
CFLAGS = -I$(SRC_DIR) -Wall -g -O0
CXXFLAGS = -I$(SRC_DIR) -std=c++14 -Wall -g -O0

# OpenCV flags
OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

# Add OpenCV flags to compilation
CFLAGS += $(OPENCV_CFLAGS)
CXXFLAGS += $(OPENCV_CFLAGS)

# Auto dependency generation for headers
DEPFLAGS = -MMD -MP

# TensorRT flags (uncomment when implementing)
# TRT_LIBS = -lnvinfer -lcudart

LDFLAGS = -lstdc++
LDFLAGS += $(OPENCV_LIBS) #$(TRT_LIBS)

# Default target
all: dirs $(TARGET)

# Create build directories
dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/vision

# Link
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile C sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# Compile C++ sources
$(BUILD_DIR)/vision/%.o: $(VISION_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Rebuild
rebuild: clean all

# Include generated dependency files when present
-include $(OBJS:.o=.d)

.PHONY: all dirs clean rebuild
