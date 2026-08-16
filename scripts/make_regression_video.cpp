#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: make_regression_video IMAGE OUTPUT_AVI [FRAME_COUNT]\n";
        return 2;
    }
    const cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "cannot read input image\n";
        return 1;
    }
    cv::VideoWriter writer(argv[2], cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           5.0, image.size(), true);
    if (!writer.isOpened()) {
        std::cerr << "cannot open regression video writer\n";
        return 1;
    }
    const int frame_count = argc == 4 ? std::atoi(argv[3]) : 8;
    if (frame_count <= 0) {
        std::cerr << "FRAME_COUNT must be positive\n";
        return 2;
    }
    for (int frame = 0; frame < frame_count; ++frame) {
        writer.write(image);
    }
    writer.release();
    return 0;
}
