/*
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-07
 */

#if !defined(__SGM_CPU_H__)
#define __SGM_CPU_H__

#include "./common/sgm.h"

namespace sgm
{
    class SGM_CPU: public SGM
    {
        private:
            uint32_t *_census_map_left;
            uint32_t *_census_map_right;
            uint16_t *_cost_map_initial;
            uint16_t *_cost_map_right;
            uint16_t *_cost_map_scanline_buffer;
            uint16_t *_cost_map_aggregated;
            uint16_t *_disparity_corse;
            uint16_t *_disparity_corse_right;
            float *_disparity_refined;

            /// @brief 初始化内部内存空间
            void initial_memory_space();
            /// @brief 释放内部内存空间
            void destroy_memory_space();

        public:
            SGM_CPU(int32_t width, int32_t height, int32_t P1, int32_t P2);
            ~SGM_CPU();

            void calculate_disparity(uint8_t* left, uint8_t* right, float* disparity);
    };

}



#endif // __SGM_CPU_H__
