#include <iostream>
#include <chrono>
#include <future>
#include <thread>
#include <exception>
#include <vector>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio/registry.hpp>
#include "opencv2/core/core_c.h"

#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "passing_one_obj.hpp"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

#define IMGSHOW_COLS 960
#define IMGSHOW_ROWS 540

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

std::vector<passing_one_obj<cv::Mat> *> frame_vec;
std::atomic<bool> exit_flag(false);

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void read_video_src(void)
{
    
    cv::VideoCapture cap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw,format=I420 ! appsink"); 

    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "error opening video source." << std::endl;
        return;
    } 

    while (!exit_flag.load()) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_YUV2BGR_I420);
        frame_vec[0]->send(bgr);
    }
    cap.release();
    return;
}


bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, float& gd, float& gw, std::string& img_dir) {
    if (std::string(argv[1]) == "-csi") {
        engine = std::string(argv[2]);
    }
    else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::string wts_name = "";
    std::string engine_name = "";
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    if (!parse_args(argc, argv, wts_name, engine_name, gd, gw, img_dir)) {
        std::cerr << "./yolov5 -csi [engine-file]       // run inference with jetson CSI camera and save result to output files." << std::endl;
        return -1;
    }

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    bool sync = false;
    std::vector<std::future<void>> future_vec;
    std::vector<cv::VideoWriter> out_file_vec;
    int grid_size = 1;
    int subimg_cols = IMGSHOW_COLS/grid_size;
    int subimg_rows = IMGSHOW_ROWS/grid_size;

    frame_vec.push_back(new passing_one_obj<cv::Mat>(sync));
    future_vec.push_back(std::async(std::launch::async, read_video_src));

    // save video files
    cv::VideoWriter out;
    out.open("csi-camera-out.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 25.0, cv::Size(subimg_cols, subimg_rows), true);
    out_file_vec.push_back(out);
        
    cv::Mat img_show[BATCH_SIZE];
    while (true) {
        int fcount = 0;
        std::vector <cv::Mat> img_display_vec;
        for (int f = 0; f < (int)future_vec.size(); f++) {
            fcount++;
            if (fcount < BATCH_SIZE && f + 1 != (int)future_vec.size()) continue;
            for (int b = 0; b < fcount; b++) {  
                cv::Mat img = frame_vec[f]->receive();
                if (img.empty()) continue;
                img_show[b] = img.clone();
                cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H); // letterbox BGR to RGB
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row) {
                    uchar* uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col) {
                        data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
            }

            // Run inference
            doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
            std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
            }
            for (int b = 0; b < fcount; b++) {
                auto& res = batch_res[b];
                for (size_t j = 0; j < res.size(); j++) {
                    cv::Rect r = get_rect(img_show[b], res[j].bbox);
                    cv::rectangle(img_show[b], r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                    cv::putText(img_show[b], std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                }
                // resize image 
                cv::resize(img_show[b], img_show[b], cv::Size(subimg_cols, subimg_rows), 0, 0, cv::INTER_AREA);
                // save to display vector
                img_display_vec.push_back(img_show[b]);
            }
            fcount = 0;
        }
        // display multiple images in a single window 
        cv::Mat img_dst(540, 960, CV_8UC3, cv::Scalar(0,50,0));
        for (int i = 0; i < (int)img_display_vec.size(); i++)   { 
            img_display_vec[i].copyTo(img_dst(cv::Rect((i%grid_size) * subimg_cols, ((i/grid_size)%grid_size) * subimg_rows, subimg_cols, subimg_rows)));
            // write video file
            out_file_vec[i].write(img_display_vec[i]);
        }
        cv::imshow("Objcet Detection Overlay", img_dst);
        if (cv::waitKey(33) == 27) {
            exit_flag.store(true);
            break;
        }
    }  
    cv::destroyWindow("Objcet Detection Overlay");
    
    for (auto i: out_file_vec) 
        i.release();
    std::cout << "videowriter released..." << std::endl;
    
    // clear frames in buffers
    for (int i = 0; i < (int)future_vec.size(); i++) {
        cv::Mat tmp;
        if (frame_vec[i]->is_object_present())
            tmp = frame_vec[i]->receive();
        future_vec[i].get();
    }

    
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
