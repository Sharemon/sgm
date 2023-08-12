/*
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-08
 */

#include "./sgm_gpu/sgm_gpu.h"
#include "./common/sgm_util.h"

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#define CUDA_CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}


/// @brief 构造函数
/// @param width 图像宽度
/// @param height 图像高度
/// @param P1 P1参数
/// @param P2 P2参数
/// @param apply_postrpocess 是否启用后处理 
sgm::SGM_GPU::SGM_GPU(int32_t width, int32_t height, int32_t P1, int32_t P2):
    SGM(width, height, P1, P2)
{
    initial_memory_space();
}

sgm::SGM_GPU::~SGM_GPU()
{
    destroy_memory_space();
}

__global__ void census_calculate_gpu(uint8_t *img_left, uint8_t *img_right, 
                                     int32_t width, int32_t height,
                                     uint32_t *census_left, uint32_t *census_right)
{
    const int32_t block_window_rows = 32 + CENSUS_WINDOW_HEIGHT/2 * 2;
    const int32_t block_window_cols = 32 + CENSUS_WINDOW_WIDTH/2 * 2;

    __shared__ uint8_t block_window_left[block_window_rows][block_window_cols];
    __shared__ uint8_t block_window_right[block_window_rows][block_window_cols];

    int32_t smid_x = (threadIdx.x + threadIdx.y * blockDim.x) % block_window_cols;
    int32_t smid_y = (threadIdx.x + threadIdx.y * blockDim.x) / block_window_cols;

    int32_t gmid_x = blockIdx.x * blockDim.x - CENSUS_WINDOW_WIDTH/2 + smid_x;
    int32_t gmid_y = blockIdx.y * blockDim.y - CENSUS_WINDOW_HEIGHT/2 + smid_y;

    bool valid = (gmid_x >= 0) && (gmid_x < width) && (gmid_y >= 0) && (gmid_y < height);

    // 赋值前一大半
    block_window_left[smid_y][smid_x]  = valid ? img_left[gmid_x + gmid_y * width] : 0;
    block_window_right[smid_y][smid_x] = valid ? img_right[gmid_x + gmid_y * width] : 0;

    // 剩余的少量数据赋值
    if (threadIdx.x + threadIdx.y * blockDim.x < block_window_rows * block_window_cols - blockDim.x * blockDim.y)
    {
        smid_x = (blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x) % block_window_cols;
        smid_y = (blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x) / block_window_cols;

        gmid_x = blockIdx.x * blockDim.x - CENSUS_WINDOW_WIDTH/2 + smid_x;
        gmid_y = blockIdx.y * blockDim.y - CENSUS_WINDOW_HEIGHT/2 + smid_y;

        valid = (gmid_x >= 0) && (gmid_x < width) && (gmid_y >= 0) && (gmid_y < height);
        block_window_left[smid_y][smid_x]  = valid ? img_left[gmid_x + gmid_y * width] : 0;
        block_window_right[smid_y][smid_x] = valid ? img_right[gmid_x + gmid_y * width] : 0;
    }
    __syncthreads();

    // census算子计算
    int32_t lid_x = threadIdx.x + CENSUS_WINDOW_WIDTH/2;
    int32_t lid_y = threadIdx.y + CENSUS_WINDOW_HEIGHT/2;

    int32_t row, col;
    uint8_t cur_left, ref_left = block_window_left[lid_y][lid_x];
    uint8_t cur_right, ref_right = block_window_right[lid_y][lid_x];
    uint32_t val_left = 0, val_right = 0;
    
    for (row = -CENSUS_WINDOW_HEIGHT/2; row <= CENSUS_WINDOW_HEIGHT/2; row++)
    {
        for (col = -CENSUS_WINDOW_WIDTH/2; col<= CENSUS_WINDOW_WIDTH/2; col++)
        {
            val_left = (val_left << 1);
            val_right = (val_right << 1);

            cur_left = block_window_left[lid_y + row][lid_x + col];
            cur_right = block_window_right[lid_y + row][lid_x + col];

            if (cur_left > ref_left)
            {
                val_left += 1;
            }

            if (cur_right > ref_right)
            {
                val_right += 1;
            }
        }
    } 

    // 赋值
    int32_t gid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t gid_y = threadIdx.y + blockIdx.y * blockDim.y;

    valid = (gid_x >= 0) && (gid_x < width) && (gid_y >= 0) && (gid_y < height);

    if (valid)
    {
        census_left[gid_x + gid_y * width] = val_left;
        census_right[gid_x + gid_y * width] = val_right;
    }
}


__global__ void census_match_gpu(uint32_t *census_left, uint32_t *census_right, int32_t _width, uint16_t *cost)
{
    extern __shared__ uint32_t shared_mem[];
    uint32_t *sm_census_left = (uint32_t *)shared_mem;
    uint32_t *sm_census_right = (uint32_t *)((uint8_t *)shared_mem + blockDim.x * sizeof(uint32_t));

    int32_t entire_block = _width / blockDim.x;
    int32_t data_len = blockIdx.x < entire_block? blockDim.x : _width - entire_block * blockDim.x;

    // 如果blockIdx.x == 0 就不填充sm_census_right前面的DISPARITY_MAX个数据了, 不太所谓 
    if (blockIdx.x != 0)
    {
        if (threadIdx.x < DISPARITY_MAX)
        {
            sm_census_right[threadIdx.x] = census_right[blockIdx.y * _width + blockDim.x * blockIdx.x + threadIdx.x - DISPARITY_MAX];
        }
    }
    
    if (threadIdx.x < data_len)
    {
        sm_census_left[threadIdx.x] = census_left[blockIdx.y * _width + blockDim.x * blockIdx.x + threadIdx.x];
        sm_census_right[threadIdx.x + DISPARITY_MAX] = census_right[blockIdx.y * _width + blockDim.x * blockIdx.x + threadIdx.x];
        
        __syncthreads();

        for (int32_t i=0; i< DISPARITY_MAX; i++)
        {
            uint32_t census_left_val = sm_census_left[threadIdx.x];
            uint32_t census_right_val = sm_census_right[threadIdx.x + DISPARITY_MAX - i];

            uint32_t census_xor = census_left_val ^ census_right_val;
            cost[i + (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * _width) * DISPARITY_MAX] = 
                threadIdx.x + blockIdx.x * blockDim.x < i? UINT8_MAX : __popc(census_xor);
        }
    }
}

/// @brief 计算视差
/// @param left 左图
/// @param right 右图
/// @param disparity 视差结果
void sgm::SGM_GPU::calculate_disparity(uint8_t* left, uint8_t* right, float* disparity)
{
    // 0.
    memcpy(_img_left, left, _width * _height * sizeof(uint8_t));
    memcpy(_img_right, right, _width * _height * sizeof(uint8_t));

    // 1.
    //census_calculate(_img_left, _width, _height, _census_map_left, CENSUS_WINDOW_WIDTH, CENSUS_WINDOW_HEIGHT);
    //census_calculate(_img_right, _width, _height, _census_map_right, CENSUS_WINDOW_WIDTH, CENSUS_WINDOW_HEIGHT);
    dim3 block(32,32);
    dim3 grid((_width-1)/block.x + 1, (_height-1)/block.y + 1);
    census_calculate_gpu<<<grid, block>>>(  _img_left, _img_right, 
                                        _width, _height, 
                                        _census_map_left, _census_map_right);
    cudaDeviceSynchronize();

    // 2. 
    //census_match(_census_map_left, _census_map_right, _width, _height, _cost_map_initial, DISPARITY_MAX);
    block.x = _width < 1024 ? _width : 1024;
    block.y = 1;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = _height; 
    census_match_gpu<<<grid, block, (DISPARITY_MAX + 2*block.x)*sizeof(uint32_t)>>>(
        _census_map_left, _census_map_right, _width, _cost_map_initial);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaGetLastError());


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
void sgm::SGM_GPU::initial_memory_space()
{
    //CUDA_CHECK(cudaHostAlloc((void **)&_img_left , _width * _height * sizeof(uint8_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocManaged((void **)&_img_left , _width * _height * sizeof(uint8_t)));
    //CUDA_CHECK(cudaHostAlloc((void **)&_img_right, _width * _height * sizeof(uint8_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocManaged((void **)&_img_right, _width * _height * sizeof(uint8_t)));
    
    //CUDA_CHECK(cudaHostAlloc((void **)&_census_map_left , _width * _height * sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocManaged((void **)&_census_map_left , _width * _height * sizeof(uint32_t)));
    //CUDA_CHECK(cudaHostAlloc((void **)&_census_map_right, _width * _height * sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocManaged((void **)&_census_map_right, _width * _height * sizeof(uint32_t)));
    
    //CUDA_CHECK(cudaHostAlloc((void **)&_cost_map_initial, _width * _height * DISPARITY_MAX * sizeof(uint16_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaMallocManaged((void **)&_cost_map_initial, _width * _height * DISPARITY_MAX * sizeof(uint16_t)));

    _cost_map_right = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint16_t));
    _cost_map_scanline_buffer = (uint16_t *)malloc(SCAN_LINE_PATH * _width * _height * DISPARITY_MAX * sizeof(uint16_t));
    _cost_map_aggregated = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint16_t));
    
    _disparity_corse = (uint16_t *)malloc(_width * _height * sizeof(uint16_t));
    _disparity_corse_right = (uint16_t *)malloc(_width * _height * sizeof(uint16_t));
    _disparity_refined = (float *)malloc(_width * _height * sizeof(float));
}

void sgm::SGM_GPU::destroy_memory_space()
{
    CUDA_CHECK(cudaFree(_img_left));
    CUDA_CHECK(cudaFree(_img_right));
    CUDA_CHECK(cudaFree(_census_map_left));
    CUDA_CHECK(cudaFree(_census_map_right));
    CUDA_CHECK(cudaFree(_cost_map_initial));
    free(_cost_map_right);
    free(_cost_map_scanline_buffer);
    free(_cost_map_aggregated);
    free(_disparity_corse);
    free(_disparity_corse_right);
    free(_disparity_refined);
}

