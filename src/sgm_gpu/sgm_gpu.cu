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

#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }

/// @brief 构造函数
/// @param width 图像宽度
/// @param height 图像高度
/// @param P1 P1参数
/// @param P2 P2参数
/// @param apply_postrpocess 是否启用后处理
sgm::SGM_GPU::SGM_GPU(int32_t width, int32_t height, int32_t P1, int32_t P2) : SGM(width, height, P1, P2)
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
    const int32_t block_window_rows = 32 + CENSUS_WINDOW_HEIGHT / 2 * 2;
    const int32_t block_window_cols = 32 + CENSUS_WINDOW_WIDTH / 2 * 2;

    __shared__ uint8_t block_window_left[block_window_rows][block_window_cols];
    __shared__ uint8_t block_window_right[block_window_rows][block_window_cols];

    int32_t smid_x = (threadIdx.x + threadIdx.y * blockDim.x) % block_window_cols;
    int32_t smid_y = (threadIdx.x + threadIdx.y * blockDim.x) / block_window_cols;

    int32_t gmid_x = blockIdx.x * blockDim.x - CENSUS_WINDOW_WIDTH / 2 + smid_x;
    int32_t gmid_y = blockIdx.y * blockDim.y - CENSUS_WINDOW_HEIGHT / 2 + smid_y;

    bool valid = (gmid_x >= 0) && (gmid_x < width) && (gmid_y >= 0) && (gmid_y < height);

    // 赋值前一大半
    block_window_left[smid_y][smid_x] = valid ? img_left[gmid_x + gmid_y * width] : 0;
    block_window_right[smid_y][smid_x] = valid ? img_right[gmid_x + gmid_y * width] : 0;

    // 剩余的少量数据赋值
    if (threadIdx.x + threadIdx.y * blockDim.x < block_window_rows * block_window_cols - blockDim.x * blockDim.y)
    {
        smid_x = (blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x) % block_window_cols;
        smid_y = (blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x) / block_window_cols;

        gmid_x = blockIdx.x * blockDim.x - CENSUS_WINDOW_WIDTH / 2 + smid_x;
        gmid_y = blockIdx.y * blockDim.y - CENSUS_WINDOW_HEIGHT / 2 + smid_y;

        valid = (gmid_x >= 0) && (gmid_x < width) && (gmid_y >= 0) && (gmid_y < height);
        block_window_left[smid_y][smid_x] = valid ? img_left[gmid_x + gmid_y * width] : 0;
        block_window_right[smid_y][smid_x] = valid ? img_right[gmid_x + gmid_y * width] : 0;
    }
    __syncthreads();

    // census算子计算
    int32_t lid_x = threadIdx.x + CENSUS_WINDOW_WIDTH / 2;
    int32_t lid_y = threadIdx.y + CENSUS_WINDOW_HEIGHT / 2;

    int32_t row, col;
    uint8_t cur_left, ref_left = block_window_left[lid_y][lid_x];
    uint8_t cur_right, ref_right = block_window_right[lid_y][lid_x];
    uint32_t val_left = 0, val_right = 0;

    for (row = -CENSUS_WINDOW_HEIGHT / 2; row <= CENSUS_WINDOW_HEIGHT / 2; row++)
    {
        for (col = -CENSUS_WINDOW_WIDTH / 2; col <= CENSUS_WINDOW_WIDTH / 2; col++)
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
    int32_t data_len = blockIdx.x < entire_block ? blockDim.x : _width - entire_block * blockDim.x;

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

        for (int32_t i = 0; i < DISPARITY_MAX; i++)
        {
            uint32_t census_left_val = sm_census_left[threadIdx.x];
            uint32_t census_right_val = sm_census_right[threadIdx.x + DISPARITY_MAX - i];

            uint32_t census_xor = census_left_val ^ census_right_val;
            cost[i + (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * _width) * DISPARITY_MAX] =
                threadIdx.x + blockIdx.x * blockDim.x < i ? UINT8_MAX : __popc(census_xor);
        }
    }
}

__global__ void WTA_gpu(uint16_t *cost, uint16_t *disparity)
{
    __shared__ uint16_t cost_of_one_pixel[DISPARITY_MAX];
    __shared__ uint16_t min_idx[DISPARITY_MAX];

    int32_t smid = threadIdx.x;
    int32_t gmid = (blockIdx.x + blockIdx.y * gridDim.x) * DISPARITY_MAX + threadIdx.x;

    cost_of_one_pixel[smid] = cost[gmid];
    min_idx[smid] = threadIdx.x;
    __syncthreads();

    // 规约比较
    int32_t tid = threadIdx.x;

    for (int32_t stride = DISPARITY_MAX / 2; stride > 0; stride = (stride >> 1))
    {
        if (tid < stride)
        {
            if (cost_of_one_pixel[tid] > cost_of_one_pixel[tid + stride])
            {
                cost_of_one_pixel[tid] = cost_of_one_pixel[tid + stride];
                min_idx[tid] = min_idx[tid + stride];
            }
        }
    }

    if (tid == 0)
    {
        disparity[blockIdx.x + blockIdx.y * gridDim.x] = min_idx[tid];
    }
}

__global__ void compute_disparity_right_gpu(uint16_t *cost, uint16_t *disparity_right)
{
    __shared__ uint16_t cost_of_one_pixel[DISPARITY_MAX];
    __shared__ uint16_t min_idx[DISPARITY_MAX];

    int32_t d = threadIdx.x;
    int32_t x = blockIdx.x;
    int32_t y = blockIdx.y;
    int32_t width = gridDim.x;

    int32_t smid = d;
    int32_t gmid = (x + d + y * width) * DISPARITY_MAX + d;

    cost_of_one_pixel[smid] = x + d < width ? cost[gmid] : UINT8_MAX;
    min_idx[smid] = threadIdx.x;
    __syncthreads();

    // 规约比较
    int32_t tid = threadIdx.x;

    for (int32_t stride = DISPARITY_MAX / 2; stride > 0; stride = (stride >> 1))
    {
        if (tid < stride)
        {
            if (cost_of_one_pixel[tid] > cost_of_one_pixel[tid + stride])
            {
                cost_of_one_pixel[tid] = cost_of_one_pixel[tid + stride];
                min_idx[tid] = min_idx[tid + stride];
            }
        }
    }

    if (tid == 0)
    {
        disparity_right[blockIdx.x + blockIdx.y * gridDim.x] = min_idx[tid];
    }
}

__global__ void LR_check_gpu(uint16_t *disparity_left, uint16_t *disparity_right, int32_t width)
{
    extern __shared__ uint16_t shared_mem_LR_check[];
    uint16_t *sm_disparity_left = (uint16_t *)shared_mem_LR_check;
    uint16_t *sm_disparity_right = (uint16_t *)((uint8_t *)shared_mem_LR_check + blockDim.x * sizeof(uint16_t));

    int32_t entire_block = width / blockDim.x;
    int32_t data_len = blockIdx.x < entire_block ? blockDim.x : width - entire_block * blockDim.x;

    // 如果blockIdx.x == 0 就不填充sm_census_right前面的DISPARITY_MAX个数据了, 不太所谓
    if (blockIdx.x != 0)
    {
        if (threadIdx.x < DISPARITY_MAX)
        {
            sm_disparity_right[threadIdx.x] = disparity_right[blockIdx.y * width + blockDim.x * blockIdx.x + threadIdx.x - DISPARITY_MAX];
        }
    }
    else
    {
        if (threadIdx.x < DISPARITY_MAX)
        {
            sm_disparity_right[threadIdx.x] = UINT16_MAX;
        }
    }

    if (threadIdx.x < data_len)
    {
        sm_disparity_left[threadIdx.x] = disparity_left[blockIdx.y * width + blockDim.x * blockIdx.x + threadIdx.x];
        sm_disparity_right[threadIdx.x + DISPARITY_MAX] = disparity_right[blockIdx.y * width + blockDim.x * blockIdx.x + threadIdx.x];

        __syncthreads();

        uint16_t disparity_left_val = sm_disparity_left[threadIdx.x];
        uint16_t disparity_right_val = sm_disparity_right[threadIdx.x + DISPARITY_MAX - disparity_left_val];

        if (abs(disparity_left_val - disparity_right_val) > 1)
        {
            disparity_left[blockIdx.y * width + blockDim.x * blockIdx.x + threadIdx.x] = 0;
        }
    }
}

__global__ void refine_gpu(uint16_t *cost, uint16_t *disparity_int, float *disparity_float, int32_t width)
{
    int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t y = blockIdx.y;

    if (x < width)
    {
        int32_t d = disparity_int[x + y * width];

        if (d != 0 && d != DISPARITY_MAX - 1)
        {
            uint16_t c0 = cost[(y * width + x) * DISPARITY_MAX + d];
            uint16_t c1 = cost[(y * width + x) * DISPARITY_MAX + d - 1];
            uint16_t c2 = cost[(y * width + x) * DISPARITY_MAX + d + 1];

            float demon = c1 + c2 - 2 * c0;
            float dsub = demon < 1 ? d : d + (c1 - c2) / demon / 2.0f;

            disparity_float[x + y * width] = dsub;
        }
        else
        {
            disparity_float[x + y * width] = d;
        }
    }
}

__global__ void median_filter3x3_gpu(float *in, float *out, int32_t width, int32_t height)
{
    int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    float val[9];
    float min;
    int32_t min_idx;

    if (x > 0 && x < width - 1 &&  y > 0 && y < height -1)
    {
        val[0] = in[(x-1) + (y-1) * width];
        val[1] = in[(x+0) + (y-1) * width];
        val[2] = in[(x+1) + (y-1) * width];
        val[3] = in[(x-1) + (y+0) * width];
        val[4] = in[(x+0) + (y+0) * width];
        val[5] = in[(x+1) + (y+0) * width];
        val[6] = in[(x-1) + (y+1) * width];
        val[7] = in[(x+0) + (y+1) * width];
        val[8] = in[(x+1) + (y+1) * width];

        // 排序只排一半就够了
        for (int32_t i = 0;i < 5; i++)
        {
            min = val[i];
            min_idx = i;
            for (int32_t j = i + 1; j < 9; j++)
            {
                if (val[j] < min)
                {
                    min = val[j];
                    min_idx = j;
                }
            }

            val[min_idx] = val[i];
            val[i] = min;
        }

        out[x + y*width] = val[4];
    }
    else
    {
        out[x + y*width] = in[x + y*width];
    }
}

__global__ void cost_accumulation_gpu(uint16_t *cost_scanline, uint16_t *cost_acc)
{
    int32_t d = threadIdx.x;
    int32_t x = blockIdx.x;
    int32_t y = blockIdx.y;
    int32_t width = gridDim.x;
    int32_t height = gridDim.y;

    uint16_t *cost_scanline_array[SCAN_LINE_PATH];
    cost_scanline_array[0] = cost_scanline;
    cost_scanline_array[1] = cost_scanline_array[0] + width * height * DISPARITY_MAX;
    cost_scanline_array[2] = cost_scanline_array[1] + width * height * DISPARITY_MAX;
    cost_scanline_array[3] = cost_scanline_array[2] + width * height * DISPARITY_MAX;
#if SCAN_LINE_PATH == 8
    cost_scanline_array[4] = cost_scanline_array[3] + width * height * DISPARITY_MAX;
    cost_scanline_array[5] = cost_scanline_array[4] + width * height * DISPARITY_MAX;
    cost_scanline_array[6] = cost_scanline_array[5] + width * height * DISPARITY_MAX;
    cost_scanline_array[7] = cost_scanline_array[6] + width * height * DISPARITY_MAX;
#endif

    int32_t idx = (x + y * width) * DISPARITY_MAX + d;
    uint32_t cost_temp_u32 =    (uint32_t)cost_scanline_array[0][idx] + 
                                (uint32_t)cost_scanline_array[1][idx] + 
                                (uint32_t)cost_scanline_array[2][idx] + 
                                (uint32_t)cost_scanline_array[3][idx];
#if SCAN_LINE_PATH == 8
    cost_temp_u32+=((uint32_t)cost_scanline_array[4][idx] +
                    (uint32_t)cost_scanline_array[5][idx] +
                    (uint32_t)cost_scanline_array[6][idx] +
                    (uint32_t)cost_scanline_array[7][idx]); 
#endif

    cost_acc[idx] = (uint16_t)(cost_temp_u32 / SCAN_LINE_PATH);
}

void cost_aggregation_gpu(  uint16_t *cost_init, uint8_t *img, uint16_t *cost_scanline, 
                            int32_t width, int32_t height, int32_t P1, int32_t P2)
{
    cudaMemcpy(cost_scanline, cost_init, width * height * DISPARITY_MAX * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(cost_scanline + width * height * DISPARITY_MAX, cost_init, width * height * DISPARITY_MAX * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(cost_scanline + width * height * DISPARITY_MAX * 2, cost_init, width * height * DISPARITY_MAX * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(cost_scanline + width * height * DISPARITY_MAX * 3, cost_init, width * height * DISPARITY_MAX * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
}

/// @brief 计算视差
/// @param left 左图
/// @param right 右图
/// @param disparity 视差结果
void sgm::SGM_GPU::calculate_disparity(uint8_t *left, uint8_t *right, float *disparity)
{
    double t0, t1;
    t0 = cpu_time_get();

    // 0.
    cudaMemcpy(_img_left_device, left, _width * _height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(_img_right_device, right, _width * _height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    t1 = cpu_time_get();
    std::cout << "memcpy time used is " << (t1 - t0) << "s" << std::endl;
    t0 = cpu_time_get();

    // 1.
    // census_calculate(_img_left, _width, _height, _census_map_left, CENSUS_WINDOW_WIDTH, CENSUS_WINDOW_HEIGHT);
    // census_calculate(_img_right, _width, _height, _census_map_right, CENSUS_WINDOW_WIDTH, CENSUS_WINDOW_HEIGHT);
    dim3 block(32, 32);
    dim3 grid((_width - 1) / block.x + 1, (_height - 1) / block.y + 1);
    census_calculate_gpu<<<grid, block>>>(_img_left_device, _img_right_device,
                                          _width, _height,
                                          _census_map_left_device, _census_map_right_device);
    cudaDeviceSynchronize();

    t1 = cpu_time_get();
    std::cout << "census calculation time used is " << (t1 - t0) << "s" << std::endl;
    t0 = cpu_time_get();

    // 2.
    // census_match(_census_map_left, _census_map_right, _width, _height, _cost_map_initial, DISPARITY_MAX);
    block.x = _width < 1024 ? _width : 1024;
    block.y = 1;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = _height;
    census_match_gpu<<<grid, block, (DISPARITY_MAX + 2 * block.x) * sizeof(uint32_t)>>>(
        _census_map_left_device, _census_map_right_device, _width, _cost_map_initial_device);
    cudaDeviceSynchronize();

    t1 = cpu_time_get();
    std::cout << "census match time used is " << (t1 - t0) << "s" << std::endl;

    t0 = cpu_time_get();

    // 3.
    // cost_aggregation(_cost_map_initial, left, _width, _height, DISPARITY_MAX,
    //                  _cost_map_aggregated, _P1, _P2, _cost_map_scanline_buffer, SCAN_LINE_PATH);
    // 3.1 cost aggregation
    cost_aggregation_gpu(_cost_map_initial_device, _img_left_device,
                         _cost_map_scanline_buffer_device, _width, _height, _P1, _P2);

    // 3.2 cost accumulation
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x = _width;
    grid.y = _height;
    cost_accumulation_gpu<<<grid,block>>>(_cost_map_scanline_buffer_device, _cost_map_aggregated_device);

    t1 = cpu_time_get();
    std::cout << "cost aggregation time used is " << (t1 - t0) << "s" << std::endl;

    t0 = cpu_time_get();

    // 4.
    // WTA(_cost_map_aggregated, _disparity_corse, _width, _height, DISPARITY_MAX);
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x = _width;
    grid.y = _height;
    WTA_gpu<<<grid, block>>>(_cost_map_aggregated_device, _disparity_corse_device);
    cudaDeviceSynchronize();

    t1 = cpu_time_get();
    std::cout << "WTA time used is " << (t1 - t0) << "s" << std::endl;

    t0 = cpu_time_get();

    // 5.
    // LR_check(_cost_map_aggregated, _disparity_corse, _cost_map_right, _disparity_corse_right, _width, _height, DISPARITY_MAX);
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x = _width;
    grid.y = _height;
    compute_disparity_right_gpu<<<grid, block>>>(_cost_map_aggregated_device, _disparity_corse_right_device);
    cudaDeviceSynchronize();

    block.x = _width < 1024 ? _width : 1024;
    block.y = 1;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = _height;
    LR_check_gpu<<<grid, block, (DISPARITY_MAX + 2 * block.x) * sizeof(uint16_t)>>>(
        _disparity_corse_device, _disparity_corse_right_device, _width);
    cudaDeviceSynchronize();

    t1 = cpu_time_get();
    std::cout << "LR_check time used is " << (t1 - t0) << "s" << std::endl;

    t0 = cpu_time_get();

    // 6.
    // refine(_cost_map_aggregated, _disparity_corse, _disparity_refined, _width, _height, DISPARITY_MAX);
    block.x = _width < 1024 ? _width : 1024;
    block.y = 1;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = _height;
    refine_gpu<<<grid, block>>>(_cost_map_aggregated_device, _disparity_corse_device, _disparity_refined_device, _width);

    t1 = cpu_time_get();
    std::cout << "refine time used is " << (t1 - t0) << "s" << std::endl;

    t0 = cpu_time_get();

    // 7.
    // median_filter(_disparity_refined, disparity, _width, _height, MEDIAN_FILTER_SIZE);
    block.x = 32;
    block.y = 32;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = (_height - 1) / block.y + 1;
    median_filter3x3_gpu<<<grid, block>>>(_disparity_refined_device, _disparity_filtered_device, _width, _height);

    t1 = cpu_time_get();
    std::cout << "median filter time used is " << (t1 - t0) << "s" << std::endl;
    t0 = cpu_time_get();

    cudaMemcpy(disparity, _disparity_filtered_device,
               _width * _height * sizeof(float), cudaMemcpyDeviceToHost);

    t1 = cpu_time_get();
    std::cout << "memcpy time used is " << (t1 - t0) << "s" << std::endl;
}

/// @brief 初始化内部内存空间
void sgm::SGM_GPU::initial_memory_space()
{
    CUDA_CHECK(cudaMalloc((void **)&_img_left_device, _width * _height * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc((void **)&_img_right_device, _width * _height * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc((void **)&_census_map_left_device, _width * _height * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **)&_census_map_right_device, _width * _height * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void **)&_cost_map_initial_device, _width * _height * DISPARITY_MAX * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **)&_cost_map_right_device, _width * _height * DISPARITY_MAX * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **)&_cost_map_scanline_buffer_device, SCAN_LINE_PATH * _width * _height * DISPARITY_MAX * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **)&_cost_map_aggregated_device, _width * _height * DISPARITY_MAX * sizeof(uint16_t)));

    CUDA_CHECK(cudaMalloc((void **)&_disparity_corse_device, _width * _height * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **)&_disparity_corse_right_device, _width * _height * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **)&_disparity_refined_device, _width * _height * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&_disparity_filtered_device, _width * _height * sizeof(float)));

    _census_map_left = (uint32_t *)malloc(_width * _height * sizeof(uint32_t));
    _census_map_right = (uint32_t *)malloc(_width * _height * sizeof(uint32_t));

    _cost_map_initial = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint16_t));
    _cost_map_right = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint16_t));
    _cost_map_scanline_buffer = (uint16_t *)malloc(SCAN_LINE_PATH * _width * _height * DISPARITY_MAX * sizeof(uint16_t));
    _cost_map_aggregated = (uint16_t *)malloc(_width * _height * DISPARITY_MAX * sizeof(uint16_t));

    _disparity_corse = (uint16_t *)malloc(_width * _height * sizeof(uint16_t));
    _disparity_corse_right = (uint16_t *)malloc(_width * _height * sizeof(uint16_t));
    _disparity_refined = (float *)malloc(_width * _height * sizeof(float));
}

void sgm::SGM_GPU::destroy_memory_space()
{
    CUDA_CHECK(cudaFree(_img_left_device));
    CUDA_CHECK(cudaFree(_img_right_device));
    CUDA_CHECK(cudaFree(_census_map_left_device));
    CUDA_CHECK(cudaFree(_census_map_right_device));
    CUDA_CHECK(cudaFree(_cost_map_initial_device));
    CUDA_CHECK(cudaFree(_cost_map_right_device));
    CUDA_CHECK(cudaFree(_cost_map_scanline_buffer_device));
    CUDA_CHECK(cudaFree(_cost_map_aggregated_device));
    CUDA_CHECK(cudaFree(_disparity_corse_device));
    CUDA_CHECK(cudaFree(_disparity_corse_right_device));
    CUDA_CHECK(cudaFree(_disparity_refined_device));
    CUDA_CHECK(cudaFree(_disparity_filtered_device));

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
