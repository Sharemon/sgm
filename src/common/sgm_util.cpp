/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-08-08
 */

#include "./common/sgm_util.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <string.h>
#include <assert.h>


/// @brief 计算图像坐标(x,y)上的census值
/// @param img 图像输入
/// @param x   坐标x
/// @param y   坐标y
/// @param census_width   census窗口宽度
/// @param census_height   census窗口高度
/// @return    census值
inline uint32_t census(const uint8_t *img, int32_t width, int32_t height,
                       int32_t x, int32_t y, int32_t census_width, int32_t census_height)
{
    uint32_t val = 0;

    int32_t x_start = std::max(0, x - (census_width - 1) / 2);
    int32_t x_end = std::min(width - 1, x + (census_width - 1) / 2);
    int32_t y_start = std::max(0, y - (census_width - 1) / 2);
    int32_t y_end = std::min(height - 1, y + (census_width + 1) / 2);

    uint8_t ref = img[x + y * width];

    for (int32_t r = y_start; r <= y_end; r++)
    {
        for (int32_t c = x_start; c <= x_end; c++)
        {
            uint8_t cur = img[c + r * width];
            if (cur > ref)
            {
                val += 1;
            }

            val = val << 1;
        }
    }

    return (val >> 1);
}

/// @brief 计算图像Census特征
/// @param img 图像数据
/// @param width 图像宽度
/// @param height 图像高度
/// @param census_map census结果
/// @param census_width census窗口宽度
/// @param census_height census窗口高度
void sgm::census_calculate(const uint8_t *img, int32_t width, int32_t height,
                           uint32_t *census_map, int32_t census_width, int32_t census_height)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            census_map[x + y * width] = census(img, width, height, x, y, census_width, census_height);
        }
    }
}

/// @brief 计算两个census的汉明距离
/// @param census1 第一个census
/// @param census2 第二个census
/// @return 汉明距离结果
inline uint16_t hanning_dist(uint32_t census1, uint32_t census2)
{
    uint32_t census_xor = census1 ^ census2;
    uint16_t dist = 0;

    while (census_xor)
    {
        dist++;
        census_xor &= (census_xor - 1);
    }

    return dist;
}

/// @brief 计算census匹配代价
/// @param census_map_left 左图census特征
/// @param census_map_right 右图census特征
/// @param width 图像宽度
/// @param height 图像高度
/// @param cost_map 代价结果
/// @param max_dispairy 最大视差
void sgm::census_match(const uint32_t *census_map_left, const uint32_t *census_map_right,
                       int32_t width, int32_t height,
                       uint16_t *cost_map, int32_t max_dispairy)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            for (int32_t i = 0; i < max_dispairy; i++)
            {
                cost_map[i + x * max_dispairy + y * width * max_dispairy] = x < i ? UINT8_MAX : hanning_dist(census_map_left[x + y * width], census_map_right[x - i + y * width]);
            }
        }
    }
}

/// @brief 从左到右代价聚合
void cost_aggregation_left2right(const uint16_t *cost_map, const uint8_t *img,
                                 int32_t width, int32_t height, int32_t max_disparity,
                                 uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);

    for (int y = 0; y < height; y++)
    {
        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(&cost_last[0], cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map += max_disparity;
        cost_aggregated += max_disparity;
        img += 1;

        for (int x = 1; x < width; x++)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(&cost_last[0], cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map += max_disparity;
            cost_aggregated += max_disparity;
            img += 1;
        }
    }
}

/// @brief 从右到左代价聚合
void cost_aggregation_right2left(const uint16_t *cost_map, const uint8_t *img,
                                 int32_t width, int32_t height, int32_t max_disparity,
                                 uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);

    cost_map += (width * height - 1) * max_disparity;
    cost_aggregated += (width * height - 1) * max_disparity;
    img += (width * height - 1);

    for (int y = height - 1; y >= 0; y--)
    {
        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map -= max_disparity;
        cost_aggregated -= max_disparity;
        img -= 1;

        for (int x = width - 2; x >= 0; x--)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map -= max_disparity;
            cost_aggregated -= max_disparity;
            img -= 1;
        }
    }
}

/// @brief 从上到下代价聚合
void cost_aggregation_up2down(const uint16_t *cost_map, const uint8_t *img,
                              int32_t width, int32_t height, int32_t max_disparity,
                              uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);

    const uint16_t *cost_org = cost_map;
    uint16_t *cost_aggre_org = cost_aggregated;
    const uint8_t *img_data_org = img;

    for (int x = 0; x < width; x++)
    {
        cost_map = &cost_org[x * max_disparity];
        cost_aggregated = &cost_aggre_org[x * max_disparity];
        img = &img_data_org[x];

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map += width * max_disparity;
        cost_aggregated += width * max_disparity;
        img += width;

        for (int y = 1; y < height; y++)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map += width * max_disparity;
            cost_aggregated += width * max_disparity;
            img += width;
        }
    }
}

/// @brief 从下到上代价聚合
void cost_aggregation_down2up(const uint16_t *cost_map, const uint8_t *img,
                              int32_t width, int32_t height, int32_t max_disparity,
                              uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);

    const uint16_t *cost_org = cost_map;
    uint16_t *cost_aggre_org = cost_aggregated;
    const uint8_t *img_data_org = img;

    for (int x = width - 1; x >= 0; x--)
    {
        cost_map = &cost_org[((height - 1) * width + x) * max_disparity];
        cost_aggregated = &cost_aggre_org[((height - 1) * width + x) * max_disparity];
        img = &img_data_org[((height - 1) * width + x)];

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map -= width * max_disparity;
        cost_aggregated -= width * max_disparity;
        img -= width;

        for (int y = height - 2; y >= 0; y--)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map -= width * max_disparity;
            cost_aggregated -= width * max_disparity;
            img -= width;
        }
    }
}

/// @brief 从左上到右下代价聚合
void cost_aggregation_leftup2rightdown(const uint16_t *cost_map, const uint8_t *img,
                                       int32_t width, int32_t height, int32_t max_disparity,
                                       uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);
    const uint16_t *cost_temp = cost_map;
    uint16_t *cost_aggre_temp = cost_aggregated;
    const uint8_t *img_data_temp = img;
    int x_cur = 0;
    int y_cur = 0;

    // 先从第一行开始聚合
    for (int x = 0; x < width; x++)
    {
        x_cur = x;
        y_cur = 0;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map += (width + 1) * max_disparity;
        cost_aggregated += (width + 1) * max_disparity;
        img += (width + 1);

        while (++x_cur < width && ++y_cur < height)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map += (width + 1) * max_disparity;
            cost_aggregated += (width + 1) * max_disparity;
            img += (width + 1);
        }
    }

    // 再从第一列开始聚合
    for (int y = 1; y < height; y++)
    {
        x_cur = 0;
        y_cur = y;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map += (width + 1) * max_disparity;
        cost_aggregated += (width + 1) * max_disparity;
        img += (width + 1);

        while (++x_cur < width && ++y_cur < height)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map += (width + 1) * max_disparity;
            cost_aggregated += (width + 1) * max_disparity;
            img += (width + 1);
        }
    }
}

/// @brief 从右下到左上代价聚合
void cost_aggregation_rightdown2leftup(const uint16_t *cost_map, const uint8_t *img,
                                       int32_t width, int32_t height, int32_t max_disparity,
                                       uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);
    const uint16_t *cost_temp = cost_map;
    uint16_t *cost_aggre_temp = cost_aggregated;
    const uint8_t *img_data_temp = img;
    int x_cur = 0;
    int y_cur = 0;

    // 先从最后一行开始聚合
    for (int x = 0; x < width; x++)
    {
        x_cur = x;
        y_cur = height - 1;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = cost_last[0];
        for (int i = 1; i < max_disparity; i++)
        {
            if (cost_min_last > cost_last[i])
            {
                cost_min_last = cost_last[i];
            }
        }

        uint8_t gray_last = (*img);

        cost_map -= (width + 1) * max_disparity;
        cost_aggregated -= (width + 1) * max_disparity;
        img -= (width + 1);

        while (--x_cur >= 0 && --y_cur >= 0)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map -= (width + 1) * max_disparity;
            cost_aggregated -= (width + 1) * max_disparity;
            img -= (width + 1);
        }
    }

    // 再从最后一列开始聚合
    for (int y = 0; y < height - 1; y++)
    {
        x_cur = width - 1;
        y_cur = y;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map -= (width + 1) * max_disparity;
        cost_aggregated -= (width + 1) * max_disparity;
        img -= (width + 1);

        while (--x_cur >= 0 && --y_cur >= 0)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map -= (width + 1) * max_disparity;
            cost_aggregated -= (width + 1) * max_disparity;
            img -= (width + 1);
        }
    }
}


/// @brief 从右上到左下代价聚合
void cost_aggregation_rightup2leftdown(const uint16_t *cost_map, const uint8_t *img,
                                       int32_t width, int32_t height, int32_t max_disparity,
                                       uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);
    const uint16_t *cost_temp = cost_map;
    uint16_t *cost_aggre_temp = cost_aggregated;
    const uint8_t *img_data_temp = img;
    int x_cur = 0;
    int y_cur = 0;

    // 先从第一行开始聚合
    for (int x = 0; x < width; x++)
    {
        x_cur = x;
        y_cur = 0;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map += (width - 1) * max_disparity;
        cost_aggregated += (width - 1) * max_disparity;
        img += (width - 1);

        while (--x_cur >= 0 && ++y_cur < height)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map += (width - 1) * max_disparity;
            cost_aggregated += (width - 1) * max_disparity;
            img += (width - 1);
        }
    }

    // 再从最后一列开始聚合
    for (int y = 1; y < height; y++)
    {
        x_cur = width - 1;
        y_cur = y;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map += (width - 1) * max_disparity;
        cost_aggregated += (width - 1) * max_disparity;
        img += (width - 1);

        while (--x_cur >= 0 && ++y_cur < height)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map += (width - 1) * max_disparity;
            cost_aggregated += (width - 1) * max_disparity;
            img += (width - 1);
        }
    }
}


/// @brief 从左下到右上代价聚合
void cost_aggregation_leftdown2rightup(const uint16_t *cost_map, const uint8_t *img,
                                       int32_t width, int32_t height, int32_t max_disparity,
                                       uint16_t *cost_aggregated, int32_t P1, int32_t P2)
{
    std::vector<uint16_t> cost_last(max_disparity);
    const uint16_t *cost_temp = cost_map;
    uint16_t *cost_aggre_temp = cost_aggregated;
    const uint8_t *img_data_temp = img;
    int x_cur = 0;
    int y_cur = 0;

    // 先从最后一行开始聚合
    for (int x = 0; x < width; x++)
    {
        x_cur = x;
        y_cur = height - 1;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        // 第一列不需聚合
        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map -= (width - 1) * max_disparity;
        cost_aggregated -= (width - 1) * max_disparity;
        img -= (width - 1);

        while (++x_cur < width && --y_cur >= 0)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map -= (width - 1) * max_disparity;
            cost_aggregated -= (width - 1) * max_disparity;
            img -= (width - 1);
        }
    }

    // 再从第一列开始聚合
    for (int y = 0; y < height - 1; y++)
    {
        x_cur = 0;
        y_cur = y;

        cost_map = cost_temp + (x_cur + y_cur * width) * max_disparity;
        cost_aggregated = cost_aggre_temp + (x_cur + y_cur * width) * max_disparity;
        img = img_data_temp + (x_cur + y_cur * width);

        // 第一列不需聚合
        memcpy(cost_aggregated, cost_map, max_disparity * sizeof(uint16_t));
        memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));

        uint16_t cost_min_last = *std::min_element(cost_last.begin(), cost_last.end());

        uint8_t gray_last = (*img);

        cost_map -= (width - 1) * max_disparity;
        cost_aggregated -= (width - 1) * max_disparity;
        img -= (width - 1);

        while (++x_cur < width && --y_cur >= 0)
        {
            uint8_t gray = (*img);
            uint16_t cost_min_cur = UINT16_MAX;
            for (int d = 0; d < max_disparity; d++)
            {
                uint16_t l0 = cost_map[d];
                uint16_t l1 = cost_last[d];
                uint16_t l2 = (d == 0 ? UINT8_MAX : cost_last[d - 1]) + P1;
                uint16_t l3 = (d == max_disparity - 1 ? UINT8_MAX : cost_last[d + 1]) + P1;
                uint16_t l4 = cost_min_last + std::max(P1, P2 / (abs(gray - gray_last) + 1));

                cost_aggregated[d] = l0 + std::min(std::min(l1, l2), std::min(l3, l4)) - cost_min_last;

                cost_min_cur = std::min(cost_aggregated[d], cost_min_cur);
            }

            memcpy(cost_last.data(), cost_aggregated, max_disparity * sizeof(uint16_t));
            cost_min_last = cost_min_cur;

            gray_last = gray;

            cost_map -= (width - 1) * max_disparity;
            cost_aggregated -= (width - 1) * max_disparity;
            img -= (width - 1);
        }
    }
}

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
void sgm::cost_aggregation(const uint16_t *cost_map, const uint8_t *img,
                           int32_t width, int32_t height, int32_t max_disparity,
                           uint16_t *cost_aggregated, int32_t P1, int32_t P2,
                           uint16_t *cost_scanline_buffer, int32_t scanline_path)
{
    assert(scanline_path == 4 || scanline_path == 8);

    uint16_t *cost_scanline[8];

    for (int32_t i = 0; i < scanline_path; i++)
    {
        cost_scanline[i] = cost_scanline_buffer + i * width * height * max_disparity;
    }

    cost_aggregation_left2right(cost_map, img, width, height, max_disparity, cost_scanline[0], P1, P2);
    cost_aggregation_right2left(cost_map, img, width, height, max_disparity, cost_scanline[1], P1, P2);
    cost_aggregation_up2down(cost_map, img, width, height, max_disparity, cost_scanline[2], P1, P2);
    cost_aggregation_down2up(cost_map, img, width, height, max_disparity, cost_scanline[3], P1, P2);

    if (scanline_path == 8)
    {
        cost_aggregation_leftup2rightdown(cost_map, img, width, height, max_disparity, cost_scanline[4], P1, P2);
        cost_aggregation_rightdown2leftup(cost_map, img, width, height, max_disparity, cost_scanline[5], P1, P2);
        cost_aggregation_rightup2leftdown(cost_map, img, width, height, max_disparity, cost_scanline[6], P1, P2);
        cost_aggregation_leftdown2rightup(cost_map, img, width, height, max_disparity, cost_scanline[7], P1, P2);
    }

    for (int32_t i = 0; i < width * height * max_disparity; i++)
    {
        uint32_t cost_i = 0;

        cost_i += (uint32_t)cost_scanline[0][i] +
                  (uint32_t)cost_scanline[1][i] +
                  (uint32_t)cost_scanline[2][i] +
                  (uint32_t)cost_scanline[3][i];

        if (scanline_path == 8)
        {
            cost_i += (uint32_t)cost_scanline[4][i] +
                      (uint32_t)cost_scanline[5][i] +
                      (uint32_t)cost_scanline[6][i] +
                      (uint32_t)cost_scanline[7][i];
        }

        cost_aggregated[i] = (uint16_t)(cost_i / scanline_path);
    }
}

/// @brief 在一个数组中找最小值对应的索引
/// @param arr 数组数据
/// @param len 数组长度
/// @return 最小值索引
inline uint32_t find_minimum_in_array(const uint16_t *arr, int len)
{
    uint32_t min_idx = 0;
    uint16_t min = arr[0];
    for (int32_t i = 1; i < len; i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
            min_idx = i;
        }
    }

    return min_idx;
}

/// @brief WTA算法
/// @param cost_map 代价图
/// @param disparity 视差图
/// @param width 图像宽度
/// @param height 图像高度
/// @param max_disparity 最大视差
void sgm::WTA(const uint16_t *cost_map, uint16_t *disparity,
              int32_t width, int32_t height, int32_t max_disparity)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            disparity[x + y * width] = find_minimum_in_array(cost_map + (x + y * width) * max_disparity, max_disparity);
        }
    }
}

/// @brief 左右一致性检查
/// @param cost_map 代价图
/// @param disparity 视差图
/// @param cost_map_r 右图代价图（输出）
/// @param disparity_r 右图视差图（输出）
/// @param width 图像宽度
/// @param height 图像高度
/// @param max_disparity 最大视差
void sgm::LR_check(const uint16_t *cost_map, uint16_t *disparity,
                   uint16_t *cost_map_r, uint16_t *disparity_r,
                   int32_t width, int32_t height, int32_t max_disparity)
{
    // 右图代价空间构建
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            for (int32_t d = 0; d < max_disparity; d++)
            {
                cost_map_r[(y * width + x) * max_disparity + d] = x + d < width ? cost_map[(y * width + x + d) * max_disparity + d] : UINT8_MAX;
            }
        }
    }

    // 求右图视差
    WTA(cost_map_r, disparity_r, width, height, max_disparity);

    // 左右一致性检查
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            uint16_t l = disparity[x + y * width];
            if (x - l < 0)
                continue;

            uint16_t r = disparity_r[x - l + y * width];
            if (abs(l - r) > 1)
            {
                disparity[x + y * width] = 0;
            }
        }
    }
}

/// @brief 视差优化
/// @param cost_map 代价图
/// @param disparity 初始视差图
/// @param disparity_float 优化后的浮点视差图
/// @param width 图像宽度
/// @param height 图像高度
/// @param max_disparity 最大视差
void sgm::refine(const uint16_t *cost_map, const uint16_t *disparity, float *disparity_float,
                 int32_t width, int32_t height, int32_t max_disparity)
{
    for (int32_t y = 0; y < height; y++)
    {
        for (int32_t x = 0; x < width; x++)
        {
            uint16_t d = disparity[x + y * width];
            if (d != 0 && d != max_disparity - 1)
            {
                uint16_t c0 = cost_map[(y * width + x) * max_disparity + d];
                uint16_t c1 = cost_map[(y * width + x) * max_disparity + d - 1];
                uint16_t c2 = cost_map[(y * width + x) * max_disparity + d + 1];

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
}

/// @brief 中值滤波
/// @param disparity_in 输入视差图
/// @param disparity_out 输出视差图
/// @param width 图像宽度
/// @param height 图像高度
/// @param kernel_size 滤波器内核大小
void sgm::median_filter(float *disparity_in, float *disparity_out, int width, int height, int32_t kernel_size)
{
    std::vector<float> win;

    for (int32_t y = kernel_size / 2; y < height - kernel_size / 2; y++)
    {
        for (int32_t x = kernel_size / 2; x < width - kernel_size / 2; x++)
        {
            win.clear();

            for (int32_t r = -kernel_size / 2; r <= kernel_size / 2; r++)
            {
                for (int32_t c = -kernel_size / 2; c <= kernel_size / 2; c++)
                {
                    win.push_back(disparity_in[x + c + (y + r) * width]);
                }
            }

            std::sort(win.begin(), win.end());
            disparity_out[x + y * width] = win[win.size() / 2];
        }
    }
}