#if !defined(__SGM_H__)
#define __SGM_H__

#include "sgm_define.h"

namespace sgm
{
    class SGM
    {
    protected:
        int _width;
        int _height;
        

        int _max_disparity;
        int _P1;
        int _P2;

        bool apply_LR_check;
        
        bool apply_median_filter;


    public:
        SGM(/* args */);
        ~SGM();
    };
    
}



#endif // __SGM_H__
