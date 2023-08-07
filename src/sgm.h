/*
 * Copyright @2023 Sharemon. All rights reserved.
 *
 * @author: sharemon
 * @date: 2023-08-07
 */

#if !defined(__SGM_H__)
#define __SGM_H__

#include "sgm_define.h"

namespace sgm
{
    class SGM
    {
    protected:
        int32_t _width;
        int32_t _height;

        int32_t _P1;
        int32_t _P2;

        bool _apply_postprocess;

    public:
        SGM(int32_t width, int32_t height, int32_t P1, int32_t P2, bool psotprocess = true): 
            _width(width), _height(height), 
            _P1(P1), _P2(P2), _apply_postprocess(psotprocess){};
        ~SGM(){};

        virtual void calculate_disparity(uint8_t *left, uint8_t* right, float *disparity) = 0;
    };
    
}



#endif // __SGM_H__
