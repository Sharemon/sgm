/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-08-08
 */

#if !defined(__SGM_UTIL_H__)
#define __SGM_UTIL_H__

#include "sgm_define.h"

namespace sgm
{
    /// @brief 计算census
    /// @param img 输入图像
    /// @param census_map 输出census map
    void census_calculate(  const uint8_t *img, int32_t width, int32_t height, 
                            uint32_t *census_map, int32_t census_width, int32_t census_height);

    /// @brief 匹配左右图像的census
    /// @param census_map_left 左图像census map
    /// @param census_map_right 右图像census map
    /// @param cost_map 代价map
    void census_match   (const uint32_t *census_map_left, const uint32_t *census_map_right, 
                        int32_t width, int32_t height, 
                        uint16_t *cost_map, int32_t max_dispairy);

    /// @brief 扫描线代价聚合
    /// @param cost_map 聚合前的代价map
    /// @param cost_aggregated 聚合后的代价map
    void cost_aggregation(  const uint16_t *cost_map, const uint8_t *img,
                            int32_t width, int32_t height, int32_t max_disparity,
                            uint16_t *cost_aggregated, int32_t P1, int32_t P2, 
                            uint16_t *cost_scanline_buffer, int32_t scanline_path);

    /// @brief winner take all, 根据代价map计算视差
    /// @param cost_map 代价map
    /// @param disparity 视差
    void WTA(   const uint16_t *cost_map, uint16_t *disparity, 
                int32_t width, int32_t height, int32_t max_disparity);

    /// @brief 左右一致性检验
    /// @param cost_map 代价map
    /// @param disparity 原始视差输入，优化后的视差输出
    void LR_check(const uint16_t *cost_map, uint16_t *disparity,
                   uint16_t *cost_map_r, uint16_t *disparity_r,
                   int32_t width, int32_t height, int32_t max_disparity);

    /// @brief 视差插值优化
    /// @param cost_map 代价map
    /// @param disparity 原始视差
    /// @param disparity_float 优化后的视差
    void refine(const uint16_t *cost_map, const uint16_t *disparity, float *disparity_float,
                int32_t width, int32_t height, int32_t max_disparity);

    /// @brief 中值滤波
    /// @param disparity 视差输入和输出
    /// @param kernel_size 滤波器内核大小
    void median_filter(float *disparity_in, float* disparity_out, int width, int height, int32_t kernel_size);
}

#endif // __SGM_UTIL_H__
