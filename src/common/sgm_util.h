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
    /// @brief 获取当前系统时间(单位:s)
    /// @return 当前系统时间
    double cpu_time_get();

    /// @brief 计算图像Census特征
    /// @param img 图像数据
    /// @param width 图像宽度
    /// @param height 图像高度
    /// @param census_map census结果
    /// @param census_width census窗口宽度
    /// @param census_height census窗口高度
    void census_calculate(  const uint8_t *img, int32_t width, int32_t height, 
                            uint32_t *census_map, int32_t census_width, int32_t census_height);

    /// @brief 计算census匹配代价
    /// @param census_map_left 左图census特征
    /// @param census_map_right 右图census特征
    /// @param width 图像宽度
    /// @param height 图像高度
    /// @param cost_map 代价结果
    /// @param max_dispairy 最大视差
    void census_match   (const uint32_t *census_map_left, const uint32_t *census_map_right, 
                        int32_t width, int32_t height, 
                        uint16_t *cost_map, int32_t max_dispairy);

    /// @brief 代价聚合
    /// @param cost_map 代价图 
    /// @param img 图像数据
    /// @param width 图像宽度
    /// @param height 图像高度
    /// @param max_disparity 最大视差
    /// @param cost_aggregated 聚合后代价
    /// @param P1 P1参数
    /// @param P2 P2参数
    /// @param cost_scanline_buffer 代价聚合工作缓存
    /// @param scanline_path 扫描线数量
    void cost_aggregation(  const uint16_t *cost_map, const uint8_t *img,
                            int32_t width, int32_t height, int32_t max_disparity,
                            uint16_t *cost_aggregated, int32_t P1, int32_t P2, 
                            uint16_t *cost_scanline_buffer, int32_t scanline_path);

    /// @brief WTA算法
    /// @param cost_map 代价图
    /// @param disparity 视差图
    /// @param width 图像宽度
    /// @param height 图像高度
    /// @param max_disparity 最大视差
    void WTA(   const uint16_t *cost_map, uint16_t *disparity, 
                int32_t width, int32_t height, int32_t max_disparity);

    
    /// @brief 左右一致性检查
    /// @param cost_map 代价图
    /// @param disparity 视差图
    /// @param cost_map_r 右图代价图（输出）
    /// @param disparity_r 右图视差图（输出）
    /// @param width 图像宽度
    /// @param height 图像高度
    /// @param max_disparity 最大视差
    void LR_check(const uint16_t *cost_map, uint16_t *disparity,
                   uint16_t *cost_map_r, uint16_t *disparity_r,
                   int32_t width, int32_t height, int32_t max_disparity);

    
    /// @brief 视差优化
    /// @param cost_map 代价图
    /// @param disparity 初始视差图
    /// @param disparity_float 优化后的浮点视差图 
    /// @param width 图像宽度
    /// @param height 图像高度
    /// @param max_disparity 最大视差
    void refine(const uint16_t *cost_map, const uint16_t *disparity, float *disparity_float,
                int32_t width, int32_t height, int32_t max_disparity);

    
    /// @brief 中值滤波
    /// @param disparity_in 输入视差图
    /// @param disparity_out 输出视差图
    /// @param width 图像宽度
    /// @param height 图像高度
    /// @param kernel_size 滤波器内核大小
    void median_filter(float *disparity_in, float* disparity_out, int width, int height, int32_t kernel_size);
}

#endif // __SGM_UTIL_H__
