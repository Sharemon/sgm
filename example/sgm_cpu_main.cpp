/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-08-08
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <../src/sgm_cpu/sgm_cpu.h>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: ./sgm_cpu [left image path] [right image path]" << std::endl;
        return EXIT_FAILURE;
    }

    // 读取数据
    std::string left_image_path = std::string(argv[1]);
    std::string right_image_path = std::string(argv[2]);
    std::string disparity_save_path = "./disparity.png";
    
    cv::Mat left_image = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

    // 将图像的长宽对齐到32bit - 4byte
    int new_img_rows = 360;// (left_image.rows + 3) / 4 * 4;
    int new_img_cols = 640;// (left_image.cols + 3) / 4 * 4;
    cv::resize(left_image, left_image, cv::Size(new_img_cols, new_img_rows));
    cv::resize(right_image, right_image, cv::Size(new_img_cols, new_img_rows));

    // 初始化sgm
    sgm::SGM_CPU sgm(left_image.cols, left_image.rows, 10, 150);

    // 初始化视差图
    cv::Mat disparity = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32FC1);

    // 计算视差图
    auto t0 = std::chrono::high_resolution_clock().now();
    sgm.calculate_disparity(left_image.data, right_image.data, (float*)disparity.data);
    auto t1 = std::chrono::high_resolution_clock().now();
    std::cout << "sgm cpu time used " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000000.0 << "s" << std::endl;

    // 展示视差图
    cv::Mat disparity_show;
    disparity.convertTo(disparity_show, CV_8UC1, 4);
    cv::imwrite(disparity_save_path, disparity_show);
    cv::imshow("result", disparity_show);
    cv::waitKey(0);

    return 0;
}