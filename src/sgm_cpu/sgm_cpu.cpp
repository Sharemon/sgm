/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-07
 */

#include "./sgm_cpu/sgm_cpu.h"
#include "./common/sgm_util.h"

#include <stdlib.h>
#include <string.h>
#include <algorithm>

/// @brief 构造函数
/// @param width 图像宽度
/// @param height 图像高度
/// @param P1 P1参数
/// @param P2 P2参数
/// @param apply_postrpocess 是否启用后处理 
sgm::SGM_CPU::SGM_CPU(int32_t width, int32_t height, int32_t P1, int32_t P2):
    SGM(width, height, P1, P2)
{
    initial_memory_space();
}

sgm::SGM_CPU::~SGM_CPU()
{
    destroy_memory_space();
}

/// @brief 计算视差
/// @param left 左图
/// @param right 右图
/// @param disparity 视差结果
void sgm::SGM_CPU::calculate_disparity(uint8_t* left, uint8_t* right, float* disparity)
{
    // 1.
    census_calculate(left, _width, _height, _census_map_left, CENSUS_WINDOW_WIDTH, CENSUS_WINDOW_HEIGHT);
    census_calculate(right, _width, _height, _census_map_right, CENSUS_WINDOW_WIDTH, CENSUS_WINDOW_HEIGHT);

    // 2. 
    census_match(_census_map_left, _census_map_right, _width, _height, _cost_map_initial, DISPARITY_MAX);

    // 3. 
    cost_aggregation(_cost_map_initial, left, _width, _height, DISPARITY_MAX, 
                     _cost_map_aggregated, _P1, _P2, _cost_map_scanline_buffer, SCAN_LINE_PATH);

    // 4. 
    WTA(_cost_map_aggregated, _disparity_corse, _width, _height, DISPARITY_MAX);

    // 5. 
    LR_check(_cost_map_aggregated, _disparity_corse, _cost_map_right, _disparity_corse_right, _width, _height, DISPARITY_MAX);

    // 6. 
    refine(_cost_map_aggregated, _disparity_corse, _disparity_refined, _width, _height, DISPARITY_MAX);

    // 7.
    median_filter(_disparity_refined, disparity, _width, _height, MEDIAN_FILTER_SIZE);
}


/// @brief 初始化内部内存空间
void sgm::SGM_CPU::initial_memory_space()
{
    _census_map_left = (uint32_t *)malloc(_width * _height * sizeof(uint32_t));
    _census_map_right = (uint32_t *)malloc(_width * _height * sizeof(uint32_t));

    _cost_map_initial = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_right = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_scanline_buffer = (uint16_t *)malloc(SCAN_LINE_PATH * _width * _height * DISPARITY_MAX * sizeof(uint32_t));
    _cost_map_aggregated = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint32_t));
    
    _disparity_corse = (uint16_t *)malloc(_width * _height * sizeof(uint16_t));
    _disparity_corse_right = (uint16_t *)malloc(_width * _height * sizeof(uint16_t));
    _disparity_refined = (float *)malloc(_width * _height * sizeof(float));
}

void sgm::SGM_CPU::destroy_memory_space()
{
    free(_census_map_left);
    free(_census_map_right);
    free(_cost_map_initial);
    free(_cost_map_right);
    free(_cost_map_scanline_buffer);
    free(_cost_map_aggregated);
    free(_disparity_corse);
    free(_disparity_corse_right);
    free(_disparity_refined);
}

