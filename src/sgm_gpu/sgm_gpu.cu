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

/// @brief cuda api返回检查
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

/// @brief 计算census特征(GPU verison)，左右图一起算
/// @param img_left 左图数据
/// @param img_right 右图数据
/// @param width 图像宽度
/// @param height 图像高度
/// @param census_left 左图census结果
/// @param census_right 右图census结果
/// @return 
__global__ void census_calculate_gpu(uint8_t *img_left, uint8_t *img_right,
                                     int32_t width, int32_t height,
                                     uint32_t *census_left, uint32_t *census_right)
{
    // 初始化共享内存
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

/// @brief census特征匹配
/// @param census_left 左图census特征
/// @param census_right 右图census特征
/// @param width 图像宽度 
/// @param cost 代价结果
/// @return 
__global__ void census_match_gpu(uint32_t *census_left, uint32_t *census_right, int32_t width, uint16_t *cost)
{
    extern __shared__ uint32_t shared_mem[];
    uint32_t *sm_census_left = (uint32_t *)shared_mem;
    uint32_t *sm_census_right = (uint32_t *)((uint8_t *)shared_mem + blockDim.x * sizeof(uint32_t));

    int32_t entire_block = width / blockDim.x;
    int32_t data_len = blockIdx.x < entire_block ? blockDim.x : width - entire_block * blockDim.x;

    // 如果blockIdx.x == 0 就不填充sm_census_right前面的DISPARITY_MAX个数据了, 不太所谓
    if (blockIdx.x != 0)
    {
        if (threadIdx.x < DISPARITY_MAX)
        {
            sm_census_right[threadIdx.x] = census_right[blockIdx.y * width + blockDim.x * blockIdx.x + threadIdx.x - DISPARITY_MAX];
        }
    }

    // census匹配
    if (threadIdx.x < data_len)
    {
        sm_census_left[threadIdx.x] = census_left[blockIdx.y * width + blockDim.x * blockIdx.x + threadIdx.x];
        sm_census_right[threadIdx.x + DISPARITY_MAX] = census_right[blockIdx.y * width + blockDim.x * blockIdx.x + threadIdx.x];

        __syncthreads();

        for (int32_t i = 0; i < DISPARITY_MAX; i++)
        {
            uint32_t census_left_val = sm_census_left[threadIdx.x];
            uint32_t census_right_val = sm_census_right[threadIdx.x + DISPARITY_MAX - i];

            uint32_t census_xor = census_left_val ^ census_right_val;
            cost[i + (threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * width) * DISPARITY_MAX] =
                threadIdx.x + blockIdx.x * blockDim.x < i ? UINT8_MAX : __popc(census_xor);
        }
    }
}


/// @brief 求取一个线程束中的最小值
/// @param val 输入数据
/// @return 最小值
__inline__ __device__ uint16_t warp_reduce_min(uint16_t val)
{
    val = min(val, __shfl_xor_sync(0xffffffff, val, 1));
    val = min(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = min(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = min(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = min(val, __shfl_xor_sync(0xffffffff, val, 16));

    return val;
}

/// @brief 垂直扫描线聚合
/// @param cost 初始代价
/// @param cost_scanline 聚合后代价
/// @param img 图像数据
/// @param P1 P1参数
/// @param P2 P2参数
/// @param height 图像高度
/// @param top2bottom 从上到下聚合还是从下到上聚合
__global__ void cost_scanline_vertical_gpu(
    uint16_t *cost, uint16_t *cost_scanline, uint8_t *img, 
    int32_t P1, int32_t P2, int32_t height, bool top2bottom)
{
    // 初始化
    const int32_t width = gridDim.x;
    const int32_t x = blockIdx.x;
    const int32_t d = threadIdx.x;
    const int32_t start = top2bottom ? 0 : height - 1;
    const int32_t step  = top2bottom ? 1 : -1;
    int32_t count = 0;

    uint16_t cost_aggr = UINT8_MAX;
    __shared__ uint16_t cost_last[2+DISPARITY_MAX];
    uint8_t gray_last = 0;
    uint8_t gray = 0;

    cost_last[d] = UINT8_MAX;
    if (d == 0 || d == 1)
    {
        cost_last[d+DISPARITY_MAX] = UINT8_MAX;
    }

    __shared__ uint16_t cost_last_min;
    __shared__ uint16_t block_reduce_min_buffer[32];
    const int32_t lane = threadIdx.x % warpSize;
    const int32_t wid = threadIdx.x / warpSize;

    if (d == 0)
    {
        cost_last_min = UINT8_MAX;
    }
    __syncthreads();

    // 代价聚合
    int32_t y = start;
    while(count < height)
    {
        gray = img[x + y * width];

        // 代价聚合公式：
        // cost_aggregation = cost[l, d] + min(cost[l-r, d-1] + P1, cost[l-r, d+1] + P1, min(cost[l-r]) + P2)
        uint16_t l0 = cost[d + DISPARITY_MAX * (x + y * width)];
        uint16_t l1 = cost_last[d + 1];
        uint16_t l2 = cost_last[d] + P1;
        uint16_t l3 = cost_last[d + 2] + P1;
        uint16_t l4 = cost_last_min + max(P1, P2 / (abs((int32_t)(gray - gray_last)) + 1));

        cost_aggr = l0 + min(min(l1,l2), min(l3, l4)) - cost_last_min;

        // 更新cost
        cost_last[d + 1] = cost_aggr;
        cost_scanline[d + DISPARITY_MAX * (x + y * width)] = cost_aggr;

        __syncthreads();

        // 找cost_last最小值
        cost_aggr = warp_reduce_min(cost_aggr);

        if (lane == 0)
        {
            block_reduce_min_buffer[wid] = cost_aggr;
        }
        __syncthreads();

        cost_aggr = (threadIdx.x < blockDim.x / warpSize) ? block_reduce_min_buffer[lane] : UINT16_MAX;

        if (wid == 0)
        {
            cost_aggr = warp_reduce_min(cost_aggr);
        }

        if (threadIdx.x == 0)
        {
            cost_last_min = cost_aggr;
        }

        __syncthreads();

        // 更新参数
        count ++;
        y += step;
        gray_last = gray;
    }
}

/// @brief 水平扫描线聚合
/// @param cost 初始代价
/// @param cost_scanline 聚合后代价
/// @param img 图像数据
/// @param P1 P1参数
/// @param P2 P2参数
/// @param width 图像宽度
/// @param top2bottom 从左到右聚合还是从下到左聚右
__global__ void cost_scanline_horizontal_gpu(
    uint16_t *cost, uint16_t *cost_scanline, uint8_t *img, 
    int32_t P1, int32_t P2, int32_t width, bool left2right)
{
    // 初始化
    const int32_t y = blockIdx.x;
    const int32_t d = threadIdx.x;
    const int32_t start = left2right ? 0 : width - 1;
    const int32_t step  = left2right ? 1 : -1;
    int32_t count = 0;

    uint16_t cost_aggr = UINT8_MAX;
    __shared__ uint16_t cost_last[2+DISPARITY_MAX];
    uint8_t gray_last = 0;
    uint8_t gray = 0;

    cost_last[d] = UINT8_MAX;
    if (d == 0 || d == 1)
    {
        cost_last[d+DISPARITY_MAX] = UINT8_MAX;
    }

    __shared__ uint16_t cost_last_min;
    __shared__ uint16_t block_reduce_min_buffer[32];
    const int32_t lane = threadIdx.x % warpSize;
    const int32_t wid = threadIdx.x / warpSize;

    if (d == 0)
    {
        cost_last_min = UINT8_MAX;
    }
    __syncthreads();

    // 代价聚合
    int32_t x = start;
    while(count < width)
    {
        gray = img[x + y * width];
        
        // 代价聚合公式：
        // cost_aggregation = cost[l, d] + min(cost[l-r, d-1] + P1, cost[l-r, d+1] + P1, min(cost[l-r]) + P2)
        uint16_t l0 = cost[d + DISPARITY_MAX * (x + y * width)];
        uint16_t l1 = cost_last[d + 1];
        uint16_t l2 = cost_last[d] + P1;
        uint16_t l3 = cost_last[d + 2] + P1;
        uint16_t l4 = cost_last_min + P2 / (abs((int32_t)(gray - gray_last)) + 1);

        cost_aggr = l0 + min(min(l1,l2), min(l3, l4)) - cost_last_min;

        // 更新cost
        cost_last[d + 1] = cost_aggr;
        cost_scanline[d + DISPARITY_MAX * (x + y * width)] = cost_aggr;

        __syncthreads();

        // 找cost_last最小值
        cost_aggr = warp_reduce_min(cost_aggr);

        if (lane == 0)
        {
            block_reduce_min_buffer[wid] = cost_aggr;
        }
        __syncthreads();

        cost_aggr = (threadIdx.x < blockDim.x / warpSize) ? block_reduce_min_buffer[lane] : UINT16_MAX;

        if (wid == 0)
        {
            cost_aggr = warp_reduce_min(cost_aggr);
        }

        if (threadIdx.x == 0)
        {
            cost_last_min = cost_aggr;
        }

        __syncthreads();

        // 更新参数
        count ++;
        x += step;
        gray_last = gray;
    }
}

/// @brief 代价聚合
/// @param cost_init 初始代价 
/// @param img 图像
/// @param cost_scanline 聚合后代价 
/// @param width 图像宽度
/// @param height 图像高度
/// @param P1 P1参数
/// @param P2 P2参数
void cost_aggregation_gpu(  uint16_t *cost_init, uint8_t *img, uint16_t *cost_scanline, 
                            int32_t width, int32_t height, int32_t P1, int32_t P2)
{
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);

    uint16_t *cost_scanline_array[DISPARITY_MAX];
    // 目前只支持scan_line = 4
    cost_scanline_array[0] = cost_scanline;
    cost_scanline_array[1] = cost_scanline_array[0] + width * height * DISPARITY_MAX;
    cost_scanline_array[2] = cost_scanline_array[1] + width * height * DISPARITY_MAX;
    cost_scanline_array[3] = cost_scanline_array[2] + width * height * DISPARITY_MAX;

    // @todo: 不同方向的聚合可以通过cuda stream并行
    // 1. top to bottom
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x  = width;
    grid.y  = 1;
    cost_scanline_vertical_gpu<<<grid, block>>>(cost_init, cost_scanline_array[0], img, P1, P2, height, true);
    cost_scanline_vertical_gpu<<<grid, block>>>(cost_init, cost_scanline_array[1], img, P1, P2, height, false);

    // 3. left to right
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x  = height;
    grid.y  = 1;
    cost_scanline_horizontal_gpu<<<grid, block>>>(cost_init, cost_scanline_array[2], img, P1, P2, width, true);
    cost_scanline_horizontal_gpu<<<grid, block>>>(cost_init, cost_scanline_array[3], img, P1, P2, width, false);
}

/// @brief 不同聚合方向的代价进行累加
/// @param cost_scanline 不同聚合方向的代价
/// @param cost_acc 累加待机
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

/// @brief WTA
/// @param cost 代价数据 
/// @param disparity 视差数据
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

/// @brief 计算右图视差
/// @param cost 左图代价
/// @param disparity_right 右图视差 
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

    // 左图视差转换为右图视差
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


/// @brief 左右一致性检查
/// @param disparity_left 左图视差 
/// @param disparity_right 右图视差
/// @param width 图像宽度
__global__ void LR_check_gpu(uint16_t *disparity_left, uint16_t *disparity_right, int32_t width)
{
    extern __shared__ uint16_t shared_mem_LR_check[];
    uint16_t *sm_disparity_left = (uint16_t *)shared_mem_LR_check;
    uint16_t *sm_disparity_right = (uint16_t *)((uint8_t *)shared_mem_LR_check + blockDim.x * sizeof(uint16_t));

    int32_t entire_block = width / blockDim.x;
    int32_t data_len = blockIdx.x < entire_block ? blockDim.x : width - entire_block * blockDim.x;

    // 数据填充
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

    // 左右比较
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

/// @brief 视差优化
/// @param cost 代价
/// @param disparity_int 初始视差 
/// @param disparity_float 优化后视差
/// @param width 图像宽度
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

/// @brief 中值滤波
/// @param in 输入
/// @param out 输出
/// @param width 图像宽度
/// @param height 图像高度
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

/// @brief 计算视差
/// @param left 左图
/// @param right 右图
/// @param disparity 视差结果
void sgm::SGM_GPU::calculate_disparity(uint8_t *left, uint8_t *right, float *disparity)
{
    // 0. 图像数据拷贝到GPU设备内存空间
    cudaMemcpy(_img_left_device, left, _width * _height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(_img_right_device, right, _width * _height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 1. 计算census特征
    dim3 block(32, 32);
    dim3 grid((_width - 1) / block.x + 1, (_height - 1) / block.y + 1);
    census_calculate_gpu<<<grid, block>>>(_img_left_device, _img_right_device,
                                          _width, _height,
                                          _census_map_left_device, _census_map_right_device);

    // 2. 特征匹配
    block.x = _width < 1024 ? _width : 1024;
    block.y = 1;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = _height;
    census_match_gpu<<<grid, block, (DISPARITY_MAX + 2 * block.x) * sizeof(uint32_t)>>>(
        _census_map_left_device, _census_map_right_device, _width, _cost_map_initial_device);

    // 3. 代价聚合
    // 3.1 cost aggregation
    cost_aggregation_gpu(_cost_map_initial_device, _img_left_device,
                         _cost_map_scanline_buffer_device, _width, _height, _P1, _P2);

    // 3.2 cost accumulation
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x = _width;
    grid.y = _height;
    cost_accumulation_gpu<<<grid,block>>>(_cost_map_scanline_buffer_device, _cost_map_aggregated_device);

    // 4. 计算视差
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x = _width;
    grid.y = _height;
    WTA_gpu<<<grid, block>>>(_cost_map_aggregated_device, _disparity_corse_device);

    // 5. 左右一致性检查
    // 5.1 计算右图视差
    block.x = DISPARITY_MAX;
    block.y = 1;
    grid.x = _width;
    grid.y = _height;
    compute_disparity_right_gpu<<<grid, block>>>(_cost_map_aggregated_device, _disparity_corse_right_device);

    // 5.2 左右检查
    block.x = _width < 1024 ? _width : 1024;
    block.y = 1;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = _height;
    LR_check_gpu<<<grid, block, (DISPARITY_MAX + 2 * block.x) * sizeof(uint16_t)>>>(
        _disparity_corse_device, _disparity_corse_right_device, _width);

    // 6. 视差优化
    block.x = _width < 1024 ? _width : 1024;
    block.y = 1;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = _height;
    refine_gpu<<<grid, block>>>(_cost_map_aggregated_device, _disparity_corse_device, _disparity_refined_device, _width);

    // 7. 中值滤波
    block.x = 32;
    block.y = 32;
    grid.x = (_width - 1) / block.x + 1;
    grid.y = (_height - 1) / block.y + 1;
    median_filter3x3_gpu<<<grid, block>>>(_disparity_refined_device, _disparity_filtered_device, _width, _height);

    // 8. 拷贝结果到CPU内存空间
    cudaMemcpy(disparity, _disparity_filtered_device,
               _width * _height * sizeof(float), cudaMemcpyDeviceToHost);
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
}
