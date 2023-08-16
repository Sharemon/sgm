/*
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-08
 */

#if !defined(__SGM_GPU_H__)
#define __SGM_GPU_H__

#include "./common/sgm.h"

namespace sgm
{
    class SGM_GPU: public SGM
    {
        private:
            uint8_t  *_img_left_device;
            uint8_t  *_img_right_device;
            uint32_t *_census_map_left_device;
            uint32_t *_census_map_right_device;
            uint16_t *_cost_map_initial_device;
            uint16_t *_cost_map_right_device;
            uint16_t *_cost_map_scanline_buffer_device;
            uint16_t *_cost_map_aggregated_device;
            uint16_t *_disparity_corse_device;
            uint16_t *_disparity_corse_right_device;
            float *_disparity_refined_device;
            float *_disparity_filtered_device;

            /// @brief 初始化内部内存空间
            void initial_memory_space();
            /// @brief 释放内部内存空间
            void destroy_memory_space();

        public:
            SGM_GPU(int32_t width, int32_t height, int32_t P1, int32_t P2);
            ~SGM_GPU();

            void calculate_disparity(uint8_t* left, uint8_t* right, float* disparity);
    };

}



#endif // __SGM_GPU_H__
