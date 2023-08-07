/*
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-07
 */

#if !defined(__SGM_CPU_H__)
#define __SGM_CPU_H__

#include "sgm.h"

namespace sgm
{
    class SGM_CPU: public SGM
    {
        private:
            uint32_t *_census_map_left;
            uint32_t *_census_map_right;
            uint16_t *_cost_map_initial;
            uint16_t *_cost_map_line[SCAN_LINE_PATH];
            uint16_t *_cost_map_aggregated;
            uint16_t *_disparity_corse;
            float    *_disparity_refined;

            /// @brief 初始化内部内存空间
            void initial_memory_space();
            /// @brief 释放内部内存空间
            void destroy_memory_space();

            /// @brief 计算census
            /// @param img 输入图像
            /// @param census_map 输出census map
            void census_calculate(uint8_t* img, uint32_t *census_map);

            /// @brief 匹配左右图像的census 
            /// @param census_map_left 左图像census map
            /// @param census_map_right 右图像census map
            /// @param cost_map 代价map
            void census_match(uint32_t* census_map_left, uint32_t* census_map_right, uint16_t* cost_map);

            /// @brief 扫描线代价聚合
            /// @param cost_map 聚合前的代价map 
            /// @param cost_aggregated 聚合后的代价map
            void cost_aggregation(uint16_t* cost_map, uint16_t* cost_aggregated);

            /// @brief 按固定路径进行代价聚合
            /// @param cost_map 聚合前的代价map
            /// @param cost_agrregated 聚合后的代价map
            void cost_aggre_up2down(uint16_t* cost_map, uint16_t* cost_aggregated);
            void cost_aggre_down2up(uint16_t* cost_map, uint16_t* cost_aggregated);
            void cost_aggre_left2right(uint16_t* cost_map, uint16_t* cost_aggregated);
            void cost_aggre_right2left(uint16_t* cost_map, uint16_t* cost_aggregated);
            void cost_aggre_leftup2rightdown(uint16_t* cost_map, uint16_t* cost_aggregated);
            void cost_aggre_rightdown2leftup(uint16_t* cost_map, uint16_t* cost_aggregated);
            void cost_aggre_rightup2leftdown(uint16_t* cost_map, uint16_t* cost_aggregated);
            void cost_aggre_leftdown2rightup(uint16_t* cost_map, uint16_t* cost_aggregated);

            /// @brief winner take all, 根据代价map计算视差
            /// @param cost_map 代价map
            /// @param disparity 视差
            void WTA(uint16_t* cost_map, uint16_t* disparity);

            /// @brief 左右一致性检验
            /// @param cost_map 代价map
            /// @param disparity 原始视差输入，优化后的视差输出
            void LR_check(uint16_t* cost_map, uint16_t *disparity);

            /// @brief 视差插值优化
            /// @param cost_map 代价map
            /// @param disparity 原始视差
            /// @param disparity_float 优化后的视差
            void refine(uint16_t *cost_map, uint16_t *disparity, float *disparity_float);

            /// @brief 中值滤波
            /// @param disparity 视差输入和输出
            /// @param kernel_size 滤波器内核大小
            void median_filter(float *disparity, int32_t kernel_size);
        public:
            SGM_CPU(int32_t width, int32_t height, int32_t P1, int32_t P2, bool apply_postrpocess = true);
            ~SGM_CPU();

            void calculate_disparity(uint8_t* left, uint8_t* right, float* disparity);
    };

}



#endif // __SGM_CPU_H__
