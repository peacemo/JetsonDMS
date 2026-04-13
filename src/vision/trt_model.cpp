#include "include/internal/trt_model.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace vision {

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

TrtLogger g_trt_logger;

bool isDynamicDims(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

std::size_t dimsVolume(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) {
        return 0;
    }

    std::size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return 0;
        }
        volume *= static_cast<std::size_t>(dims.d[i]);
    }
    return volume;
}

std::size_t dataTypeSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(std::uint16_t);
        case nvinfer1::DataType::kINT8:
            return sizeof(std::int8_t);
        case nvinfer1::DataType::kINT32:
            return sizeof(std::int32_t);
        default:
            return 0;
    }
}

bool checkCuda(cudaError_t error, const char* stage) {
    if (error != cudaSuccess) {
        std::cerr << "[TrtModel] CUDA error at " << stage << ": "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

// IEEE754 float32 -> float16 conversion (round to nearest).
std::uint16_t floatToHalfBits(float value) {
    union {
        float f;
        std::uint32_t u;
    } in;
    in.f = value;

    const std::uint32_t sign = (in.u >> 16U) & 0x8000U;
    std::uint32_t mantissa = in.u & 0x007fffffU;
    int exp = static_cast<int>((in.u >> 23U) & 0xffU) - 127 + 15;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000U) >> static_cast<std::uint32_t>(1 - exp);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000U) >> 13U));
    }

    if (exp >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }

    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10U) |
                                      ((mantissa + 0x00001000U) >> 13U));
}

float halfBitsToFloat(std::uint16_t half) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(half & 0x8000U)) << 16U;
    std::uint32_t exp = (half >> 10U) & 0x1fU;
    std::uint32_t mantissa = half & 0x03ffU;

    std::uint32_t out = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --exp;
            }
            mantissa &= 0x03ffU;
            out = sign | ((exp + (127U - 15U)) << 23U) | (mantissa << 13U);
        }
    } else if (exp == 31U) {
        out = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        out = sign | ((exp + (127U - 15U)) << 23U) | (mantissa << 13U);
    }

    union {
        std::uint32_t u;
        float f;
    } result;
    result.u = out;
    return result.f;
}

} // namespace

TrtModel::TrtModel()
    : is_initialized_(false),
      frame_counter_(0),
      runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      stream_(nullptr),
      input_data_type_(nvinfer1::DataType::kFLOAT),
      output_data_type_(nvinfer1::DataType::kFLOAT),
      input_is_dynamic_(false),
      input_elements_(0),
      output_elements_(0),
      input_bytes_(0),
      output_bytes_(0),
      input_device_buffer_(nullptr),
      output_device_buffer_(nullptr) {
}

TrtModel::~TrtModel() {
    release();
}

bool TrtModel::init(const std::string& model_path) {
    release();

    if (model_path.empty()) {
        std::cerr << "[TrtModel] Model path is empty" << std::endl;
        return false;
    }

    model_path_ = model_path;
    std::cout << "[TrtModel] Loading model: " << model_path << std::endl;

    std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[TrtModel] Failed to open model file: " << model_path_ << std::endl;
        return false;
    }

    const std::streamsize file_size = file.tellg();
    if (file_size <= 0) {
        std::cerr << "[TrtModel] Invalid model file size" << std::endl;
        return false;
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> model_data(static_cast<std::size_t>(file_size));
    if (!file.read(model_data.data(), file_size)) {
        std::cerr << "[TrtModel] Failed to read model file" << std::endl;
        return false;
    }

    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) {
        std::cerr << "[TrtModel] Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(model_data.data(), model_data.size());
    if (!engine_) {
        std::cerr << "[TrtModel] Failed to deserialize TensorRT engine" << std::endl;
        release();
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "[TrtModel] Failed to create execution context" << std::endl;
        release();
        return false;
    }

    const int nb_tensors = engine_->getNbIOTensors();
    if (nb_tensors <= 1) {
        std::cerr << "[TrtModel] Invalid number of io tensors: " << nb_tensors << std::endl;
        release();
        return false;
    }

    for (int i = 0; i < nb_tensors; ++i) {
        const char* tensor_name = engine_->getIOTensorName(i);
        if (tensor_name == nullptr) {
            continue;
        }

        const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensor_name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (!input_tensor_name_.empty()) {
                std::cerr << "[TrtModel] Multiple input tensors are not supported" << std::endl;
                release();
                return false;
            }
            input_tensor_name_ = tensor_name;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (output_tensor_name_.empty()) {
                output_tensor_name_ = tensor_name;
            }
        }
    }

    if (input_tensor_name_.empty() || output_tensor_name_.empty()) {
        std::cerr << "[TrtModel] Failed to identify input/output tensors" << std::endl;
        release();
        return false;
    }

    input_data_type_ = engine_->getTensorDataType(input_tensor_name_.c_str());
    output_data_type_ = engine_->getTensorDataType(output_tensor_name_.c_str());
    input_dims_ = engine_->getTensorShape(input_tensor_name_.c_str());
    output_dims_ = engine_->getTensorShape(output_tensor_name_.c_str());
    input_is_dynamic_ = isDynamicDims(input_dims_);

    if (!checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate")) {
        release();
        return false;
    }

    if (!input_is_dynamic_) {
        input_elements_ = dimsVolume(input_dims_);
        output_elements_ = dimsVolume(output_dims_);

        const std::size_t input_type_size = dataTypeSize(input_data_type_);
        const std::size_t output_type_size = dataTypeSize(output_data_type_);
        if (input_elements_ == 0 || output_elements_ == 0 ||
            input_type_size == 0 || output_type_size == 0) {
            std::cerr << "[TrtModel] Unsupported tensor shape or data type" << std::endl;
            release();
            return false;
        }

        input_bytes_ = input_elements_ * input_type_size;
        output_bytes_ = output_elements_ * output_type_size;

        if (!checkCuda(cudaMalloc(&input_device_buffer_, input_bytes_), "cudaMalloc(input)")) {
            release();
            return false;
        }
        if (!checkCuda(cudaMalloc(&output_device_buffer_, output_bytes_), "cudaMalloc(output)")) {
            release();
            return false;
        }

        if (!context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_)) {
            std::cerr << "[TrtModel] Failed to bind input tensor address" << std::endl;
            release();
            return false;
        }
        if (!context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_)) {
            std::cerr << "[TrtModel] Failed to bind output tensor address" << std::endl;
            release();
            return false;
        }

        if (input_data_type_ == nvinfer1::DataType::kFLOAT) {
            input_host_f32_.resize(input_elements_);
        } else if (input_data_type_ == nvinfer1::DataType::kHALF) {
            input_host_f16_.resize(input_elements_);
        } else {
            std::cerr << "[TrtModel] Unsupported input dtype" << std::endl;
            release();
            return false;
        }

        if (output_data_type_ == nvinfer1::DataType::kFLOAT) {
            output_host_f32_.resize(output_elements_);
        } else if (output_data_type_ == nvinfer1::DataType::kHALF) {
            output_host_f16_.resize(output_elements_);
        } else {
            std::cerr << "[TrtModel] Unsupported output dtype" << std::endl;
            release();
            return false;
        }
    }

    is_initialized_ = true;
    std::cout << "[TrtModel] Engine initialized successfully" << std::endl;
    return is_initialized_;
}

bool TrtModel::infer(const cv::Mat& image, DetectResult& result) {
    if (!is_initialized_) {
        std::cerr << "[TrtModel] Engine not initialized" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        result.frame_id = -1;
        return false;
    }

    if (image.empty()) {
        std::cerr << "[TrtModel] Input image is empty" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        result.frame_id = -1;
        return false;
    }

    frame_counter_++;
    result.frame_id = frame_counter_;

    if (!convertToTensorRT(image)) {
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        return false;
    }

    if (!context_->enqueueV3(stream_)) {
        std::cerr << "[TrtModel] enqueueV3 failed" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        return false;
    }

    if (output_data_type_ == nvinfer1::DataType::kFLOAT) {
        if (!checkCuda(cudaMemcpyAsync(output_host_f32_.data(),
                                       output_device_buffer_,
                                       output_bytes_,
                                       cudaMemcpyDeviceToHost,
                                       stream_),
                       "cudaMemcpyAsync(output float)")) {
            result.status = DETECT_STATUS_ERROR;
            result.confidence = 0.0f;
            return false;
        }
    } else if (output_data_type_ == nvinfer1::DataType::kHALF) {
        if (!checkCuda(cudaMemcpyAsync(output_host_f16_.data(),
                                       output_device_buffer_,
                                       output_bytes_,
                                       cudaMemcpyDeviceToHost,
                                       stream_),
                       "cudaMemcpyAsync(output half)")) {
            result.status = DETECT_STATUS_ERROR;
            result.confidence = 0.0f;
            return false;
        }
    } else {
        std::cerr << "[TrtModel] Unsupported output dtype" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        return false;
    }

    if (!checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize")) {
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        return false;
    }

    if (!processOutput(result)) {
        return false;
    }

    return true;
}

void TrtModel::release() {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
    }

    if (input_device_buffer_ != nullptr) {
        cudaFree(input_device_buffer_);
        input_device_buffer_ = nullptr;
    }
    if (output_device_buffer_ != nullptr) {
        cudaFree(output_device_buffer_);
        output_device_buffer_ = nullptr;
    }

    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    if (context_ != nullptr) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_ != nullptr) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_ != nullptr) {
        delete runtime_;
        runtime_ = nullptr;
    }

    input_tensor_name_.clear();
    output_tensor_name_.clear();
    input_host_f32_.clear();
    input_host_f16_.clear();
    output_host_f32_.clear();
    output_host_f16_.clear();

    input_dims_ = nvinfer1::Dims{};
    output_dims_ = nvinfer1::Dims{};
    input_elements_ = 0;
    output_elements_ = 0;
    input_bytes_ = 0;
    output_bytes_ = 0;
    input_is_dynamic_ = false;
    is_initialized_ = false;
}

bool TrtModel::convertToTensorRT(const cv::Mat& image) {
    if (image.channels() != 3) {
        std::cerr << "[TrtModel] Expected 3-channel image, got "
                  << image.channels() << std::endl;
        return false;
    }

    cv::Mat input_fp32;
    if (image.type() == CV_32FC3) {
        input_fp32 = image;
    } else {
        image.convertTo(input_fp32, CV_32FC3);
    }

    if (!input_fp32.isContinuous()) {
        input_fp32 = input_fp32.clone();
    }

    nvinfer1::Dims effective_input_dims = input_dims_;
    if (input_is_dynamic_) {
        if (effective_input_dims.nbDims != 3 && effective_input_dims.nbDims != 4) {
            std::cerr << "[TrtModel] Unsupported dynamic input rank: "
                      << effective_input_dims.nbDims << std::endl;
            return false;
        }

        if (effective_input_dims.nbDims == 4) {
            effective_input_dims.d[0] = 1;
            effective_input_dims.d[1] = 3;
            effective_input_dims.d[2] = input_fp32.rows;
            effective_input_dims.d[3] = input_fp32.cols;
        } else {
            effective_input_dims.d[0] = 3;
            effective_input_dims.d[1] = input_fp32.rows;
            effective_input_dims.d[2] = input_fp32.cols;
        }

        if (!context_->setInputShape(input_tensor_name_.c_str(), effective_input_dims)) {
            std::cerr << "[TrtModel] Failed to set dynamic input dimensions" << std::endl;
            return false;
        }
        if (!context_->allInputDimensionsSpecified()) {
            std::cerr << "[TrtModel] Dynamic input dimensions are not fully specified" << std::endl;
            return false;
        }

        input_dims_ = context_->getTensorShape(input_tensor_name_.c_str());
        output_dims_ = context_->getTensorShape(output_tensor_name_.c_str());

        input_elements_ = dimsVolume(input_dims_);
        output_elements_ = dimsVolume(output_dims_);
        const std::size_t input_type_size = dataTypeSize(input_data_type_);
        const std::size_t output_type_size = dataTypeSize(output_data_type_);
        if (input_elements_ == 0 || output_elements_ == 0 ||
            input_type_size == 0 || output_type_size == 0) {
            std::cerr << "[TrtModel] Failed to resolve dynamic tensor shape" << std::endl;
            return false;
        }

        input_bytes_ = input_elements_ * input_type_size;
        output_bytes_ = output_elements_ * output_type_size;

        if (input_device_buffer_ != nullptr) {
            cudaFree(input_device_buffer_);
            input_device_buffer_ = nullptr;
        }
        if (output_device_buffer_ != nullptr) {
            cudaFree(output_device_buffer_);
            output_device_buffer_ = nullptr;
        }
        if (!checkCuda(cudaMalloc(&input_device_buffer_, input_bytes_), "cudaMalloc(input dynamic)")) {
            return false;
        }
        if (!checkCuda(cudaMalloc(&output_device_buffer_, output_bytes_), "cudaMalloc(output dynamic)")) {
            return false;
        }

        if (!context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_)) {
            std::cerr << "[TrtModel] Failed to bind dynamic input buffer" << std::endl;
            return false;
        }
        if (!context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_)) {
            std::cerr << "[TrtModel] Failed to bind dynamic output buffer" << std::endl;
            return false;
        }

        if (input_data_type_ == nvinfer1::DataType::kFLOAT) {
            input_host_f32_.resize(input_elements_);
            input_host_f16_.clear();
        } else if (input_data_type_ == nvinfer1::DataType::kHALF) {
            input_host_f16_.resize(input_elements_);
            input_host_f32_.clear();
        }

        if (output_data_type_ == nvinfer1::DataType::kFLOAT) {
            output_host_f32_.resize(output_elements_);
            output_host_f16_.clear();
        } else if (output_data_type_ == nvinfer1::DataType::kHALF) {
            output_host_f16_.resize(output_elements_);
            output_host_f32_.clear();
        }
    }

    int expected_c = 0;
    int expected_h = 0;
    int expected_w = 0;
    if (input_dims_.nbDims == 4) {
        expected_c = input_dims_.d[1];
        expected_h = input_dims_.d[2];
        expected_w = input_dims_.d[3];
    } else if (input_dims_.nbDims == 3) {
        expected_c = input_dims_.d[0];
        expected_h = input_dims_.d[1];
        expected_w = input_dims_.d[2];
    } else {
        std::cerr << "[TrtModel] Unsupported input rank: " << input_dims_.nbDims << std::endl;
        return false;
    }

    if (expected_c != 3 || expected_h != input_fp32.rows || expected_w != input_fp32.cols) {
        std::cerr << "[TrtModel] Input shape mismatch, engine expects CxHxW="
                  << expected_c << "x" << expected_h << "x" << expected_w
                  << " but got 3x" << input_fp32.rows << "x" << input_fp32.cols << std::endl;
        return false;
    }

    if (input_elements_ != static_cast<std::size_t>(expected_c * expected_h * expected_w)) {
        std::cerr << "[TrtModel] Unexpected input element count" << std::endl;
        return false;
    }

    const float* src = reinterpret_cast<const float*>(input_fp32.data);
    const int hw = expected_h * expected_w;

    if (input_data_type_ == nvinfer1::DataType::kFLOAT) {
        for (int c = 0; c < expected_c; ++c) {
            float* dst_channel = input_host_f32_.data() + static_cast<std::size_t>(c) * hw;
            for (int i = 0; i < hw; ++i) {
                dst_channel[i] = src[i * expected_c + c];
            }
        }
        if (!checkCuda(cudaMemcpyAsync(input_device_buffer_,
                                       input_host_f32_.data(),
                                       input_bytes_,
                                       cudaMemcpyHostToDevice,
                                       stream_),
                       "cudaMemcpyAsync(input float)")) {
            return false;
        }
        return true;
    }

    if (input_data_type_ == nvinfer1::DataType::kHALF) {
        for (int c = 0; c < expected_c; ++c) {
            std::uint16_t* dst_channel = input_host_f16_.data() + static_cast<std::size_t>(c) * hw;
            for (int i = 0; i < hw; ++i) {
                dst_channel[i] = floatToHalfBits(src[i * expected_c + c]);
            }
        }
        if (!checkCuda(cudaMemcpyAsync(input_device_buffer_,
                                       input_host_f16_.data(),
                                       input_bytes_,
                                       cudaMemcpyHostToDevice,
                                       stream_),
                       "cudaMemcpyAsync(input half)")) {
            return false;
        }
        return true;
    }

    std::cerr << "[TrtModel] Unsupported input dtype" << std::endl;
    return false;
}

bool TrtModel::processOutput(DetectResult& result) {
    if (output_elements_ < 2) {
        std::cerr << "[TrtModel] Output elements < 2, cannot parse binary classifier" << std::endl;
        result.status = DETECT_STATUS_UNKNOWN;
        result.confidence = 0.0f;
        return false;
    }

    // User-defined class order: 0 = drowsy, 1 = non-drowsy.
    float score_drowsy = 0.0f;
    float score_non_drowsy = 0.0f;

    if (output_data_type_ == nvinfer1::DataType::kFLOAT) {
        score_drowsy = output_host_f32_[0];
        score_non_drowsy = output_host_f32_[1];
    } else if (output_data_type_ == nvinfer1::DataType::kHALF) {
        score_drowsy = halfBitsToFloat(output_host_f16_[0]);
        score_non_drowsy = halfBitsToFloat(output_host_f16_[1]);
    } else {
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        return false;
    }

    if (!std::isfinite(score_non_drowsy) || !std::isfinite(score_drowsy)) {
        std::cerr << "[TrtModel] Non-finite output score" << std::endl;
        result.status = DETECT_STATUS_ERROR;
        result.confidence = 0.0f;
        return false;
    }

    float prob_non_drowsy = 0.0f;
    float prob_drowsy = 0.0f;
    const float score_sum = score_non_drowsy + score_drowsy;
    if (score_non_drowsy >= 0.0f && score_non_drowsy <= 1.0f &&
        score_drowsy >= 0.0f && score_drowsy <= 1.0f &&
        score_sum > 0.0f && score_sum <= 1.2f) {
        // Already probability-like output.
        prob_non_drowsy = score_non_drowsy;
        prob_drowsy = score_drowsy;
    } else {
        const float max_score = std::max(score_non_drowsy, score_drowsy);
        const float exp_non_drowsy = std::exp(score_non_drowsy - max_score);
        const float exp_drowsy = std::exp(score_drowsy - max_score);
        const float exp_sum = exp_non_drowsy + exp_drowsy;
        if (exp_sum <= std::numeric_limits<float>::epsilon()) {
            result.status = DETECT_STATUS_ERROR;
            result.confidence = 0.0f;
            return false;
        }
        prob_non_drowsy = exp_non_drowsy / exp_sum;
        prob_drowsy = exp_drowsy / exp_sum;
    }

    if (prob_drowsy >= prob_non_drowsy) {
        result.status = DETECT_STATUS_DROWSY;
        result.confidence = std::max(0.0f, std::min(1.0f, prob_drowsy));
    } else {
        result.status = DETECT_STATUS_NORMAL;
        result.confidence = std::max(0.0f, std::min(1.0f, prob_non_drowsy));
    }

    return true;
}

} // namespace vision
