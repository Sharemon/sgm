/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-08-08
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/time.h>

#include "./sgm_cpu/sgm_cpu.h"


inline double cpu_time_get()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (double)(tv.tv_sec + tv.tv_usec / 1000000.0);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: ./sgm_cpu [left image path] [right image path]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string left_image_path = std::string(argv[1]);
    std::string right_image_path = std::string(argv[2]);
    std::string disparity_save_path = "./disparity.png";
    
    cv::Mat left_image = cv::imread(left_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right_image = cv::imread(right_image_path, cv::IMREAD_GRAYSCALE);

    sgm::SGM_CPU sgm(left_image.cols, left_image.rows, 10, 150);

    cv::Mat disparity = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32FC1);

    double t0 = cpu_time_get();
    sgm.calculate_disparity(left_image.data, right_image.data, (float *)disparity.data);
    double t1 = cpu_time_get();
    std::cout << "sgm cpu time used " << t1 - t0 << "s" << std::endl;

    cv::Mat disparity_show;
    disparity.convertTo(disparity_show, CV_8UC1, 4);
    cv::imwrite(disparity_save_path, disparity_show);

    return 0;
}