/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-07
 */

#include "sgm_cpu.h"

#include <stdlib.h>

using namespace sgm;

/// @brief 构造函数
/// @param width 图像宽度
/// @param height 图像高度
/// @param P1 P1参数
/// @param P2 P2参数
/// @param apply_postrpocess 是否启用后处理 
SGM_CPU::SGM_CPU(int32_t width, int32_t height, int32_t P1, int32_t P2, bool apply_postrpocess = true):
    SGM(width, height, P1, P2, apply_postrpocess)
{
    initial_memory_space();
}

SGM_CPU::~SGM_CPU()
{
    destroy_memory_space();
}

/// @brief 计算视差
/// @param left 左图
/// @param right 右图
/// @param disparity 视差结果
void SGM_CPU::calculate_disparity(uint8_t* left, uint8_t* right, float* disparity)
{
    // 1.
    census_calculate(left, _census_map_left);
    census_calculate(right, _census_map_right);

    // 2. 
    census_match(_census_map_left, _census_map_right, _cost_map_initial);

    // 3. 
    cost_aggregation(_cost_map_initial, _cost_map_aggregated);

    // 4. 
    WTA(_cost_map_aggregated, _disparity_corse);

    // 5. 
    LR_check(_cost_map_aggregated, _disparity_corse);

    // 6. 
    refine(_cost_map_aggregated, _disparity_corse, _disparity_refined);

    // 7.
    median_filter(_disparity_refined, MEDIAN_FILTER_SIZE);
}


/// @brief 初始化内部内存空间
void SGM_CPU::initial_memory_space()
{
    _census_map_left = (uint32_t *)malloc(_width * _height * sizeof(uint32_t));
    _census_map_right = (uint32_t *)malloc(_width * _height * sizeof(uint32_t));

    _cost_map_initial = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_line[0] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_line[1] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_line[2] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_line[3] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
#if SCAN_LINE_PATH == 8
    _cost_map_line[5] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_line[6] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_line[7] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_line[8] = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
#endif
    _cost_map_aggregated = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    
    _disparity_corse = (uint16_t *)malloc(_width * _height * sizeof(uint16_t));
    _disparity_refined = (float *)malloc(_width * _height * sizeof(float));
}

void SGM_CPU::destroy_memory_space()
{
    free(_census_map_left);
    free(_census_map_right);
    free(_cost_map_initial);
    free(_cost_map_line[0]);
    free(_cost_map_line[1]);
    free(_cost_map_line[2]);
    free(_cost_map_line[3]);
#if SCAN_LINE_PATH == 8
    free(_cost_map_line[4]);
    free(_cost_map_line[5]);
    free(_cost_map_line[6]);
    free(_cost_map_line[7]);
#endif
    free(_cost_map_aggregated);
    free(_disparity_corse);
    free(_disparity_refined);
}


/// @brief 计算census
/// @param img 输入图像
/// @param census_map 输出census map
void census_calculate(uint8_t *img, uint32_t *census_map);

/// @brief 匹配左右图像的census
/// @param census_map_left 左图像census map
/// @param census_map_right 右图像census map
/// @param cost_map 代价map
void census_match(uint32_t *census_map_left, uint32_t *census_map_right, uint16_t *cost_map);

/// @brief 扫描线代价聚合
/// @param cost_map 聚合前的代价map
/// @param cost_aggregated 聚合后的代价map
void cost_aggregation(uint16_t *cost_map, uint16_t *cost_aggregated);

/// @brief 按固定路径进行代价聚合
/// @param cost_map 聚合前的代价map
/// @param cost_agrregated 聚合后的代价map
void cost_aggre_up2down(uint16_t *cost_map, uint16_t *cost_aggregated);
void cost_aggre_down2up(uint16_t *cost_map, uint16_t *cost_aggregated);
void cost_aggre_left2right(uint16_t *cost_map, uint16_t *cost_aggregated);
void cost_aggre_right2left(uint16_t *cost_map, uint16_t *cost_aggregated);
void cost_aggre_leftup2rightdown(uint16_t *cost_map, uint16_t *cost_aggregated);
void cost_aggre_rightdown2leftup(uint16_t *cost_map, uint16_t *cost_aggregated);
void cost_aggre_rightup2leftdown(uint16_t *cost_map, uint16_t *cost_aggregated);
void cost_aggre_leftdown2rightup(uint16_t *cost_map, uint16_t *cost_aggregated);

/// @brief winner take all, 根据代价map计算视差
/// @param cost_map 代价map
/// @param disparity 视差
void WTA(uint16_t *cost_map, uint16_t *disparity);

/// @brief 左右一致性检验
/// @param cost_map 代价map
/// @param disparity 原始视差输入，优化后的视差输出
void LR_check(uint16_t *cost_map, uint16_t *disparity);

/// @brief 视差插值优化
/// @param cost_map 代价map
/// @param disparity 原始视差
/// @param disparity_float 优化后的视差
void refine(uint16_t *cost_map, uint16_t *disparity, float *disparity_float);

/// @brief 中值滤波
/// @param disparity 视差输入和输出
/// @param kernel_size 滤波器内核大小
void median_filter(float *disparity, int32_t kernel_size);
