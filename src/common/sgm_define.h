/*
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-07
 */

#if !defined(__SGM_DEFINE_H__)
#define __SGM_DEFINE_H__

#include <cstdint>

// sgm algorithm parameter
#define IMAGE_WIDTH_MAX         (1280)  // 最大图像宽度
#define IMAGE_HEIGHT_MAX        (720)   // 最大图像高度
#define DISPARITY_MAX           (64)    // 最大视差范围
#define SCAN_LINE_PATH          (4)     // 扫描线数量（4 or 8）
#define CENSUS_WINDOW_WIDTH     (5)     // census特征窗口宽度
#define CENSUS_WINDOW_HEIGHT    (5)     // census特征窗口高度
#define MEDIAN_FILTER_SIZE      (3)     // 中值滤波器尺寸


#endif // __SGM_DEFINE_H__

