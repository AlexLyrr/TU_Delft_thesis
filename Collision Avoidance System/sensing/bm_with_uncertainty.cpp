// M*//////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// M*/

/****************************************************************************************\
*    Very fast SAD-based (Sum-of-Absolute-Diffrences) stereo correspondence algorithm.   *
*    Contributed by Kurt Konolige                                                        *
\****************************************************************************************/

/****************************************************************************************\
*    Uncertainty map embedded in BM.                                                          *
*    Author: Alexios Lyrakis                                                      *
\****************************************************************************************/

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include <limits>
#include <stdio.h>
//#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/cv_cpu_helper.h"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <string>

using namespace cv;
// #define SEQUENTIAL
#define UNCERTAINTY
#define PRINT_TO_FILES
// #define ORIGINAL_STEREO
// #define WRITE_IMAGE
// #define SGBM_TRUTH
// #define DOWNSAMPLE

// Declare variables
long long badPixels = 0, goodPixels = 0, totalPixels = 0;
double avg_dispError = 0;
// Declare files
std::ofstream ml_csv, ml_all_csv, sgbm_truth_csv;
std::ifstream differences_csv, predictions_csv, predictions_all_csv;
// Declare ML vectors
std::vector<cv::Point> v_coordinates;
std::vector<cv::Point> v_coordinates_all;
// Declare Images
const int height_tmp = 376, width_tmp = 1241, block_size_tmp = 9, numberOfDisparities = 80;

int min_disparity = INT32_MAX;

Mat disparityTruth(height_tmp, width_tmp, CV_8UC1, Scalar(0));
Mat sgbmTruth(height_tmp, width_tmp, CV_8UC1, Scalar(0));
Mat uncertainty_map(height_tmp, width_tmp, CV_8UC1, Scalar(0));
Mat disp_unfiltered(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat disp8_left_right_diff(height_tmp, width_tmp, CV_8UC1, Scalar(0));

// Declare ML maps
Mat m_minsad(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat m_average_sad(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat m_mind(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat m_distance_first_second(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat m_distance_first_third(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat m_distance_first_fourth(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat m_ratio_first_second(height_tmp, width_tmp, CV_32FC1, Scalar(0));
Mat m_ratio_first_third(height_tmp, width_tmp, CV_32FC1, Scalar(0));
Mat m_ratio_first_fourth(height_tmp, width_tmp, CV_32FC1, Scalar(0));
Mat m_ratio_left(height_tmp, width_tmp, CV_32FC1, Scalar(0));
Mat m_ratio_right(height_tmp, width_tmp, CV_32FC1, Scalar(0));
Mat m_disp_error(height_tmp, width_tmp, CV_32SC1, Scalar(255));
Mat m_sgbm_error(height_tmp, width_tmp, CV_32SC1, Scalar(255));
Mat m_valid(height_tmp, width_tmp, CV_32SC1, Scalar(0));

Mat m_test(height_tmp, width_tmp, CV_32SC1, Scalar(0));
Mat m_disp_error2(height_tmp, width_tmp, CV_32SC1, Scalar(0));

namespace cvAlex {

    enum { PREFILTER_NORMALIZED_RESPONSE = 0,
           PREFILTER_XSOBEL = 1 };

    struct StereoBMParams {
        StereoBMParams(int _numDisparities = 64, int _SADWindowSize = 21) {
            preFilterType = StereoBM::PREFILTER_XSOBEL;
            preFilterSize = 9;
            preFilterCap = 31;
            SADWindowSize = _SADWindowSize;
            minDisparity = 0;
            numDisparities = _numDisparities > 0 ? _numDisparities : 64;
            textureThreshold = 10;
            uniquenessRatio = 15;
            speckleRange = speckleWindowSize = 0;
            roi1 = roi2 = Rect(0, 0, 0, 0);
            disp12MaxDiff = -1;
            dispType = CV_16S;
        }

        int preFilterType;
        int preFilterSize;
        int preFilterCap;
        int SADWindowSize;
        int minDisparity;
        int numDisparities;
        int textureThreshold;
        int uniquenessRatio;
        int speckleRange;
        int speckleWindowSize;
        Rect roi1, roi2;
        int disp12MaxDiff;
        int dispType;
    };

    static void prefilterNorm(const Mat &src, Mat &dst, int winsize, int ftzero, uchar *buf) {
        int x, y, wsz2 = winsize / 2;
        int *vsum = (int *)alignPtr(buf + (wsz2 + 1) * sizeof(vsum[0]), 32);
        int scale_g = winsize * winsize / 8, scale_s = (1024 + scale_g) / (scale_g * 2);
        const int OFS = 256 * 5, TABSZ = OFS * 2 + 256;
        uchar tab[TABSZ];
        const uchar *sptr = src.ptr();
        int srcstep = (int)src.step;
        Size size = src.size();

        scale_g *= scale_s;

        for (x = 0; x < TABSZ; x++)
            tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);

        for (x = 0; x < size.width; x++)
            vsum[x] = (ushort)(sptr[x] * (wsz2 + 2));

        for (y = 1; y < wsz2; y++) {
            for (x = 0; x < size.width; x++)
                vsum[x] = (ushort)(vsum[x] + sptr[srcstep * y + x]);
        }

        for (y = 0; y < size.height; y++) {
            const uchar *top = sptr + srcstep * MAX(y - wsz2 - 1, 0);
            const uchar *bottom = sptr + srcstep * MIN(y + wsz2, size.height - 1);
            const uchar *prev = sptr + srcstep * MAX(y - 1, 0);
            const uchar *curr = sptr + srcstep * y;
            const uchar *next = sptr + srcstep * MIN(y + 1, size.height - 1);
            uchar *dptr = dst.ptr<uchar>(y);

            for (x = 0; x < size.width; x++)
                vsum[x] = (ushort)(vsum[x] + bottom[x] - top[x]);

            for (x = 0; x <= wsz2; x++) {
                vsum[-x - 1] = vsum[0];
                vsum[size.width + x] = vsum[size.width - 1];
            }

            int sum = vsum[0] * (wsz2 + 1);
            for (x = 1; x <= wsz2; x++)
                sum += vsum[x];

            int val = ((curr[0] * 5 + curr[1] + prev[0] + next[0]) * scale_g - sum * scale_s) >> 10;
            dptr[0] = tab[val + OFS];

            for (x = 1; x < size.width - 1; x++) {
                sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
                val = ((curr[x] * 4 + curr[x - 1] + curr[x + 1] + prev[x] + next[x]) * scale_g - sum * scale_s) >> 10;
                dptr[x] = tab[val + OFS];
            }

            sum += vsum[x + wsz2] - vsum[x - wsz2 - 1];
            val = ((curr[x] * 5 + curr[x - 1] + prev[x] + next[x]) * scale_g - sum * scale_s) >> 10;
            dptr[x] = tab[val + OFS];
        }
    }

    static void prefilterXSobel(const Mat &src, Mat &dst, int ftzero) {
        int x, y;
        const int OFS = 256 * 4, TABSZ = OFS * 2 + 256;
        uchar tab[TABSZ] = {0};
        Size size = src.size();

        for (x = 0; x < TABSZ; x++)
            tab[x] = (uchar)(x - OFS < -ftzero ? 0 : x - OFS > ftzero ? ftzero * 2 : x - OFS + ftzero);
        uchar val0 = tab[0 + OFS];

        for (y = 0; y < size.height - 1; y += 2) {
            const uchar *srow1 = src.ptr<uchar>(y);
            const uchar *srow0 = y > 0 ? srow1 - src.step : size.height > 1 ? srow1 + src.step : srow1;
            const uchar *srow2 = y < size.height - 1 ? srow1 + src.step : size.height > 1 ? srow1 - src.step : srow1;
            const uchar *srow3 = y < size.height - 2 ? srow1 + src.step * 2 : srow1;
            uchar *dptr0 = dst.ptr<uchar>(y);
            uchar *dptr1 = dptr0 + dst.step;

            dptr0[0] = dptr0[size.width - 1] = dptr1[0] = dptr1[size.width - 1] = val0;
            x = 1;

#if CV_SIMD128
            {
                v_int16x8 ftz = v_setall_s16((short)ftzero);
                v_int16x8 ftz2 = v_setall_s16((short)(ftzero * 2));
                v_int16x8 z = v_setzero_s16();

                for (; x <= (size.width - 1) - 8; x += 8) {
                    v_int16x8 s00 = v_reinterpret_as_s16(v_load_expand(srow0 + x + 1));
                    v_int16x8 s01 = v_reinterpret_as_s16(v_load_expand(srow0 + x - 1));
                    v_int16x8 s10 = v_reinterpret_as_s16(v_load_expand(srow1 + x + 1));
                    v_int16x8 s11 = v_reinterpret_as_s16(v_load_expand(srow1 + x - 1));
                    v_int16x8 s20 = v_reinterpret_as_s16(v_load_expand(srow2 + x + 1));
                    v_int16x8 s21 = v_reinterpret_as_s16(v_load_expand(srow2 + x - 1));
                    v_int16x8 s30 = v_reinterpret_as_s16(v_load_expand(srow3 + x + 1));
                    v_int16x8 s31 = v_reinterpret_as_s16(v_load_expand(srow3 + x - 1));

                    v_int16x8 d0 = s00 - s01;
                    v_int16x8 d1 = s10 - s11;
                    v_int16x8 d2 = s20 - s21;
                    v_int16x8 d3 = s30 - s31;

                    v_uint16x8 v0 = v_reinterpret_as_u16(v_max(v_min(d0 + d1 + d1 + d2 + ftz, ftz2), z));
                    v_uint16x8 v1 = v_reinterpret_as_u16(v_max(v_min(d1 + d2 + d2 + d3 + ftz, ftz2), z));

                    v_pack_store(dptr0 + x, v0);
                    v_pack_store(dptr1 + x, v1);
                }
            }
#endif

            for (; x < size.width - 1; x++) {
                int d0 = srow0[x + 1] - srow0[x - 1], d1 = srow1[x + 1] - srow1[x - 1],
                    d2 = srow2[x + 1] - srow2[x - 1], d3 = srow3[x + 1] - srow3[x - 1];
                int v0 = tab[d0 + d1 * 2 + d2 + OFS];
                int v1 = tab[d1 + d2 * 2 + d3 + OFS];
                dptr0[x] = (uchar)v0;
                dptr1[x] = (uchar)v1;
            }
        }

        for (; y < size.height; y++) {
            uchar *dptr = dst.ptr<uchar>(y);
            x = 0;
#if CV_SIMD128
            {
                v_uint8x16 val0_16 = v_setall_u8(val0);
                for (; x <= size.width - 16; x += 16)
                    v_store(dptr + x, val0_16);
            }
#endif
            for (; x < size.width; x++)
                dptr[x] = val0;
        }
    }

    static const int DISPARITY_SHIFT_16S = 4;
    static const int DISPARITY_SHIFT_32S = 8;

    template <typename T>
    struct dispShiftTemplate {};

    template <>
    struct dispShiftTemplate<short> {
        enum { value = DISPARITY_SHIFT_16S };
    };

    template <>
    struct dispShiftTemplate<int> {
        enum { value = DISPARITY_SHIFT_32S };
    };

    template <typename T>
    inline T dispDescale(int /*v1*/, int /*v2*/, int /*d*/);

    template <>
    inline short dispDescale(int v1, int v2, int d) {
        return (short)((v1 * 256 + (d != 0 ? v2 * 256 / d : 0) + 15) >> 4);
    }

    template <>
    inline int dispDescale(int v1, int v2, int d) {
        return (int)(v1 * 256 + (d != 0 ? v2 * 256 / d : 0)); // no need to add 127, this will be converted to float
    }

#if CV_SIMD128
    template <typename dType>
    static void findStereoCorrespondenceBM_SIMD(const Mat &left, const Mat &right, Mat &disp, Mat &cost,
                                                StereoBMParams &state, uchar *buf, int _dy0, int _dy1, int row_start,
                                                int row_end) {
        int block_size_tmp;
        if (row_start == 0) {
            block_size_tmp = 9;
        } else {
            block_size_tmp = 0;
        }

        const int ALIGN = 16;
        int x, y, d;
        int wsz = state.SADWindowSize, wsz2 = wsz / 2;
        int dy0 = MIN(_dy0, wsz2 + 1), dy1 = MIN(_dy1, wsz2 + 1);
        int ndisp = state.numDisparities;
        int mindisp = state.minDisparity;
        int lofs = MAX(ndisp - 1 + mindisp, 0);
        int rofs = -MIN(ndisp - 1 + mindisp, 0);
        int width = left.cols, height = left.rows;
        int width1 = width - rofs - ndisp + 1;
        int ftzero = state.preFilterCap;
        int textureThreshold = state.textureThreshold;
        int uniquenessRatio = state.uniquenessRatio;
        const int disp_shift = dispShiftTemplate<dType>::value;
        dType FILTERED = (dType)((mindisp - 1) << disp_shift);

        ushort *sad, *hsad0, *hsad, *hsad_sub;
        int *htext;
        uchar *cbuf0, *cbuf;
        const uchar *lptr0 = left.ptr() + lofs;
        const uchar *rptr0 = right.ptr() + rofs;
        const uchar *lptr, *lptr_sub, *rptr;
        dType *dptr = disp.ptr<dType>();
        int sstep = (int)left.step;
        int dstep = (int)(disp.step / sizeof(dptr[0]));
        int cstep = (height + dy0 + dy1) * ndisp;
        short costbuf = 0;
        int coststep = cost.data ? (int)(cost.step / sizeof(costbuf)) : 0;
        const int TABSZ = 256;
        uchar tab[TABSZ];
        const v_int16x8 d0_8 = v_int16x8(0, 1, 2, 3, 4, 5, 6, 7), dd_8 = v_setall_s16(8);

        sad = (ushort *)alignPtr(buf + sizeof(sad[0]), ALIGN);
        hsad0 = (ushort *)alignPtr(sad + ndisp + 1 + dy0 * ndisp, ALIGN);
        htext = (int *)alignPtr((int *)(hsad0 + (height + dy1) * ndisp) + wsz2 + 2, ALIGN);
        cbuf0 = (uchar *)alignPtr((uchar *)(htext + height + wsz2 + 2) + dy0 * ndisp, ALIGN);

        for (x = 0; x < TABSZ; x++)
            tab[x] = (uchar)std::abs(x - ftzero);

        // initialize buffers
        memset(hsad0 - dy0 * ndisp, 0, (height + dy0 + dy1) * ndisp * sizeof(hsad0[0]));
        memset(htext - wsz2 - 1, 0, (height + wsz + 1) * sizeof(htext[0]));

        for (x = -wsz2 - 1; x < wsz2; x++) {
            hsad = hsad0 - dy0 * ndisp;
            cbuf = cbuf0 + (x + wsz2 + 1) * cstep - dy0 * ndisp;
            lptr = lptr0 + MIN(MAX(x, -lofs), width - lofs - 1) - dy0 * sstep;
            rptr = rptr0 + MIN(MAX(x, -rofs), width - rofs - ndisp) - dy0 * sstep;

            for (y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep) {
                int lval = lptr[0];
                v_uint8x16 lv = v_setall_u8((uchar)lval);
                for (d = 0; d < ndisp; d += 16) {
                    v_uint8x16 rv = v_load(rptr + d);
                    v_uint16x8 hsad_l = v_load(hsad + d);
                    v_uint16x8 hsad_h = v_load(hsad + d + 8);
                    v_uint8x16 diff = v_absdiff(lv, rv);
                    v_store(cbuf + d, diff);
                    v_uint16x8 diff0, diff1;
                    v_expand(diff, diff0, diff1);
                    hsad_l += diff0;
                    hsad_h += diff1;
                    v_store(hsad + d, hsad_l);
                    v_store(hsad + d + 8, hsad_h);
                }
                htext[y] += tab[lval];
            }
        }

        // initialize the left and right borders of the disparity map
        for (y = 0; y < height; y++) {
            for (x = 0; x < lofs; x++)
                dptr[y * dstep + x] = FILTERED;
            for (x = lofs + width1; x < width; x++)
                dptr[y * dstep + x] = FILTERED;
        }
        dptr += lofs;
        // std::cout << "Height is: " << height << "\nWidth is: " << width1 << "\nNumber of disparities is " << lofs <<
        // "\n Borders are " << dy1 << " " << dy0 << " " << wsz2 << std::endl; // @Alex

        for (x = 0; x < width1; x++, dptr++) {
            short *costptr = cost.data ? cost.ptr<short>() + lofs + x : &costbuf;
            int x0 = x - wsz2 - 1, x1 = x + wsz2;
            const uchar *cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1)) * cstep - dy0 * ndisp;
            cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1)) * cstep - dy0 * ndisp;
            hsad = hsad0 - dy0 * ndisp;
            lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width - 1 - lofs) - dy0 * sstep;
            lptr = lptr0 + MIN(MAX(x1, -lofs), width - 1 - lofs) - dy0 * sstep;
            rptr = rptr0 + MIN(MAX(x1, -rofs), width - ndisp - rofs) - dy0 * sstep;

            for (y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp, hsad += ndisp, lptr += sstep,
                lptr_sub += sstep, rptr += sstep) {
                int lval = lptr[0];
                v_uint8x16 lv = v_setall_u8((uchar)lval);
                for (d = 0; d < ndisp; d += 16) {
                    v_uint8x16 rv = v_load(rptr + d);
                    v_uint16x8 hsad_l = v_load(hsad + d);
                    v_uint16x8 hsad_h = v_load(hsad + d + 8);
                    v_uint8x16 cbs = v_load(cbuf_sub + d);
                    v_uint8x16 diff = v_absdiff(lv, rv);
                    v_int16x8 diff_l, diff_h, cbs_l, cbs_h;
                    v_store(cbuf + d, diff);
                    v_expand(v_reinterpret_as_s8(diff), diff_l, diff_h);
                    v_expand(v_reinterpret_as_s8(cbs), cbs_l, cbs_h);
                    diff_l -= cbs_l;
                    diff_h -= cbs_h;
                    hsad_h = v_reinterpret_as_u16(v_reinterpret_as_s16(hsad_h) + diff_h);
                    hsad_l = v_reinterpret_as_u16(v_reinterpret_as_s16(hsad_l) + diff_l);
                    v_store(hsad + d, hsad_l);
                    v_store(hsad + d + 8, hsad_h);
                }
                htext[y] += tab[lval] - tab[lptr_sub[0]];
            }

            // fill borders
            for (y = dy1; y <= wsz2; y++)
                htext[height + y] = htext[height + dy1 - 1];
            for (y = -wsz2 - 1; y < -dy0; y++)
                htext[y] = htext[-dy0];

            // initialize sums
            for (d = 0; d < ndisp; d++)
                sad[d] = (ushort)(hsad0[d - ndisp * dy0] * (wsz2 + 2 - dy0));

            hsad = hsad0 + (1 - dy0) * ndisp;
            for (y = 1 - dy0; y < wsz2; y++, hsad += ndisp)
                for (d = 0; d <= ndisp - 16; d += 16) {
                    v_uint16x8 s0 = v_load(sad + d);
                    v_uint16x8 s1 = v_load(sad + d + 8);
                    v_uint16x8 t0 = v_load(hsad + d);
                    v_uint16x8 t1 = v_load(hsad + d + 8);
                    s0 = s0 + t0;
                    s1 = s1 + t1;
                    v_store(sad + d, s0);
                    v_store(sad + d + 8, s1);
                }
            int tsum = 0;
            for (y = -wsz2 - 1; y < wsz2; y++)
                tsum += htext[y];

            // finally, start the real processing
            for (y = 0; y < height; y++) {

                long long averageSad = 0; // @Alex

                int minsad = INT_MAX, mind = -1;
                hsad = hsad0 + MIN(y + wsz2, height + dy1 - 1) * ndisp;
                hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0) * ndisp;
                v_int16x8 minsad8 = v_setall_s16(SHRT_MAX);
                v_int16x8 mind8 = v_setall_s16(0), d8 = d0_8;

                for (d = 0; d < ndisp; d += 16) {
                    v_int16x8 u0 = v_reinterpret_as_s16(v_load(hsad_sub + d));
                    v_int16x8 u1 = v_reinterpret_as_s16(v_load(hsad + d));

                    v_int16x8 v0 = v_reinterpret_as_s16(v_load(hsad_sub + d + 8));
                    v_int16x8 v1 = v_reinterpret_as_s16(v_load(hsad + d + 8));

                    v_int16x8 usad8 = v_reinterpret_as_s16(v_load(sad + d));
                    v_int16x8 vsad8 = v_reinterpret_as_s16(v_load(sad + d + 8));

                    u1 -= u0;
                    v1 -= v0;
                    usad8 += u1;
                    vsad8 += v1;

                    v_int16x8 mask = minsad8 > usad8;
                    minsad8 = v_min(minsad8, usad8);
                    mind8 = v_max(mind8, (mask & d8));

                    v_store(sad + d, v_reinterpret_as_u16(usad8));
                    v_store(sad + d + 8, v_reinterpret_as_u16(vsad8));

                    mask = minsad8 > vsad8;
                    minsad8 = v_min(minsad8, vsad8);

                    d8 = d8 + dd_8;
                    mind8 = v_max(mind8, (mask & d8));
                    d8 = d8 + dd_8;

                    int sum1 = v_reduce_sum(usad8); // @Alex
                    int sum2 = v_reduce_sum(vsad8); // @Alex
                    averageSad += sum1 + sum2;
                }

                tsum += htext[y + wsz2] - htext[y - wsz2 - 1];
                if (tsum < textureThreshold) {
                    dptr[y * dstep] = FILTERED;
                    continue;
                }

                ushort CV_DECL_ALIGNED(16) minsad_buf[8], mind_buf[8];
                v_store(minsad_buf, v_reinterpret_as_u16(minsad8));
                v_store(mind_buf, v_reinterpret_as_u16(mind8));
                for (d = 0; d < 8; d++)
                    if (minsad > (int)minsad_buf[d] || (minsad == (int)minsad_buf[d] && mind > mind_buf[d])) {
                        minsad = minsad_buf[d];
                        mind = mind_buf[d];
                    }

                // @Alex: Here start my own modifications
                averageSad /= ndisp;

                int second_mind = -1, third_mind = -1, fourth_mind = -1;
                int second_minsad = INT_MAX, third_minsad = INT_MAX, fourth_minsad = INT_MAX;
                double ratio_first_second = 0, ratio_first_third = 0, ratio_first_fourth, ratio_left = 0,
                       ratio_right = 0;
                int distance_first_second = 0, distance_first_third, distance_first_fourth;
#ifdef UNCERTAINTY
                for (int d = 0; d < ndisp; d++) {
                    if (d != mind) {
                        if (sad[d] < second_minsad) {
                            fourth_minsad = third_minsad;
                            third_minsad = second_minsad;
                            fourth_mind = third_mind;
                            third_mind = second_mind;
                            second_minsad = sad[d];
                            second_mind = d;
                        } else if (sad[d] < third_minsad) {
                            fourth_minsad = third_minsad;
                            fourth_mind = third_mind;
                            third_minsad = sad[d];
                            third_mind = d;
                        } else if (sad[d] < fourth_minsad) {
                            fourth_minsad = sad[d];
                            fourth_mind = d;
                        }
                    }
                }

                // std::cout << "(x,y) = (" << x + ndisp - 1 << "," << row_start + y + block_size_tmp/2 << ")" <<
                // std::endl; This 4 depends on the block size
#ifdef DOWNSAMPLE
                int dispError = ((ndisp - mind - 1) -
                                 disparityTruth.at<uchar>(row_start + y + block_size_tmp / 2, x + ndisp - 1) / 2);
#else
                int dispError =
                    ((ndisp - mind - 1) - disparityTruth.at<uchar>(row_start + y + block_size_tmp / 2, x + ndisp - 1));
#endif

#ifdef SGBM_TRUTH
                int sgbmError =
                    ((ndisp - mind - 1) - sgbmTruth.at<uchar>(row_start + y + block_size_tmp / 2, x + ndisp - 1));
#endif

                if (ndisp - mind - 1 != 0 && ndisp - mind - 1 < min_disparity) {
                    min_disparity = ndisp - mind - 1;
                }

                // std::cout << "row start " << row_start << std::endl;
                if (ml_csv.is_open() && ml_all_csv.is_open()) {
                    if (minsad != 0) { // Prevent division with 0

                        m_valid.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = 1;

                        ratio_first_second = ((double)second_minsad / (double)minsad);
                        ratio_first_third = ((double)third_minsad / (double)minsad);
                        ratio_first_fourth = ((double)fourth_minsad / (double)minsad);
                        if (mind > 0 && minsad != sad[mind - 1])
                            ratio_left = ((double)sad[mind - 1] / (double)minsad);
                        else
                            ratio_left = 1;
                        if (mind < ndisp - 1 && minsad != sad[mind + 1])
                            ratio_right = ((double)sad[mind + 1] / (double)minsad);
                        else
                            ratio_right = 1;
                        distance_first_second = ((ndisp - mind) - (ndisp - second_mind));
                        distance_first_third = ((ndisp - mind) - (ndisp - third_mind));
                        distance_first_fourth = ((ndisp - mind) - (ndisp - fourth_mind));

                        if (disparityTruth.at<uchar>(row_start + y + block_size_tmp / 2, x + ndisp - 1) != 0) {

                            // Disparity Error
                            m_disp_error.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = dispError;
                            m_disp_error2.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = dispError;

#ifdef SGBM_TRUTH
                            m_sgbm_error.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = sgbmError;
#endif
                            // Unfortunately when we run threads in parallel things gonna get messes up a little.
                            // We should have as many badPixels buffers as the number of threads and eventually
                            // sum them up.
                            if (abs(dispError) > 2) {
                                badPixels++;
                            }
                            goodPixels++;
                            avg_dispError += abs(dispError);
                        }

                        m_minsad.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = minsad;
                        m_average_sad.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = averageSad;
                        m_ratio_first_second.at<float>(row_start + y + block_size_tmp / 2, x + ndisp - 1) =
                            ratio_first_second;
                        m_ratio_first_third.at<float>(row_start + y + block_size_tmp / 2, x + ndisp - 1) =
                            ratio_first_third;
                        m_ratio_first_fourth.at<float>(row_start + y + block_size_tmp / 2, x + ndisp - 1) =
                            ratio_first_fourth;
                        m_distance_first_second.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) =
                            distance_first_second;
                        m_distance_first_third.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) =
                            distance_first_third;
                        m_distance_first_fourth.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) =
                            distance_first_fourth;
                        m_ratio_left.at<float>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = ratio_left;
                        m_ratio_right.at<float>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = ratio_right;
                    }
                }
                disp_unfiltered.at<int>(row_start + y + block_size_tmp / 2, x + ndisp - 1) = (ndisp - mind);
#endif
                if (uniquenessRatio > 0) {
                    int thresh = minsad + (minsad * uniquenessRatio / 100);
                    v_int32x4 thresh4 = v_setall_s32(thresh + 1);
                    v_int32x4 d1 = v_setall_s32(mind - 1), d2 = v_setall_s32(mind + 1);
                    v_int32x4 dd_4 = v_setall_s32(4);
                    v_int32x4 d4 = v_int32x4(0, 1, 2, 3);
                    v_int32x4 mask4;

                    for (d = 0; d < ndisp; d += 8) {
                        v_int16x8 sad8 = v_reinterpret_as_s16(v_load(sad + d));
                        v_int32x4 sad4_l, sad4_h;
                        v_expand(sad8, sad4_l, sad4_h);
                        mask4 = thresh4 > sad4_l;
                        mask4 = mask4 & ((d1 > d4) | (d4 > d2));
                        if (v_check_any(mask4))
                            break;
                        d4 += dd_4;
                        mask4 = thresh4 > sad4_h;
                        mask4 = mask4 & ((d1 > d4) | (d4 > d2));
                        if (v_check_any(mask4))
                            break;
                        d4 += dd_4;
                    }
                    if (d < ndisp) {
                        dptr[y * dstep] = FILTERED;
                        continue;
                    }
                }

                if (0 < mind && mind < ndisp - 1) {
                    int p = sad[mind + 1], n = sad[mind - 1];
                    d = p + n - 2 * sad[mind] + std::abs(p - n);
                    dptr[y * dstep] = dispDescale<dType>(ndisp - mind - 1 + mindisp, p - n, d);
                } else
                    dptr[y * dstep] = dispDescale<dType>(ndisp - mind - 1 + mindisp, 0, 0);
                costptr[y * coststep] = sad[mind];
            }
        }
    }
#endif

    /*
        Warning: This function does not implement ucnertainty map. Use SIMD instead.
     */
    template <typename mType>
    static void findStereoCorrespondenceBM(const Mat &left, const Mat &right, Mat &disp, Mat &cost,
                                           const StereoBMParams &state, uchar *buf, int _dy0, int _dy1, int row_start,
                                           int row_end) {
        // printf("%u\n", (unsigned)getThreadNum());
        // std::cout << "row start = " << row_start << " row end is " << row_end << std::endl;

        const int ALIGN = 16;
        int x, y, d;
        int wsz = state.SADWindowSize, wsz2 = wsz / 2;
        int dy0 = MIN(_dy0, wsz2 + 1), dy1 = MIN(_dy1, wsz2 + 1);
        int ndisp = state.numDisparities;
        int mindisp = state.minDisparity;
        int lofs = MAX(ndisp - 1 + mindisp, 0);
        int rofs = -MIN(ndisp - 1 + mindisp, 0);
        int width = left.cols, height = left.rows;
        int width1 = width - rofs - ndisp + 1;
        int ftzero = state.preFilterCap;
        int textureThreshold = state.textureThreshold;
        int uniquenessRatio = state.uniquenessRatio;
        const int disp_shift = dispShiftTemplate<mType>::value;
        mType FILTERED = (mType)((mindisp - 1) << disp_shift);

        int *sad, *hsad0, *hsad, *hsad_sub, *htext;
        uchar *cbuf0, *cbuf;
        const uchar *lptr0 = left.ptr() + lofs;
        const uchar *rptr0 = right.ptr() + rofs;
        const uchar *lptr, *lptr_sub, *rptr;
        mType *dptr = disp.ptr<mType>();
        int sstep = (int)left.step;
        int dstep = (int)(disp.step / sizeof(dptr[0]));
        int cstep = (height + dy0 + dy1) * ndisp;
        int costbuf = 0;
        int coststep = cost.data ? (int)(cost.step / sizeof(costbuf)) : 0;
        const int TABSZ = 256;
        uchar tab[TABSZ];

        sad = (int *)alignPtr(buf + sizeof(sad[0]), ALIGN);
        hsad0 = (int *)alignPtr(sad + ndisp + 1 + dy0 * ndisp, ALIGN);
        htext = (int *)alignPtr((int *)(hsad0 + (height + dy1) * ndisp) + wsz2 + 2, ALIGN);
        cbuf0 = (uchar *)alignPtr((uchar *)(htext + height + wsz2 + 2) + dy0 * ndisp, ALIGN);

        for (x = 0; x < TABSZ; x++)
            tab[x] = (uchar)std::abs(x - ftzero);

        // initialize buffers
        memset(hsad0 - dy0 * ndisp, 0, (height + dy0 + dy1) * ndisp * sizeof(hsad0[0]));
        memset(htext - wsz2 - 1, 0, (height + wsz + 1) * sizeof(htext[0]));

        for (x = -wsz2 - 1; x < wsz2; x++) {
            hsad = hsad0 - dy0 * ndisp;
            cbuf = cbuf0 + (x + wsz2 + 1) * cstep - dy0 * ndisp;
            lptr = lptr0 + std::min(std::max(x, -lofs), width - lofs - 1) - dy0 * sstep;
            rptr = rptr0 + std::min(std::max(x, -rofs), width - rofs - ndisp) - dy0 * sstep;
            for (y = -dy0; y < height + dy1; y++, hsad += ndisp, cbuf += ndisp, lptr += sstep, rptr += sstep) {
                int lval = lptr[0];
                d = 0;

                for (; d < ndisp; d++) {
                    int diff = std::abs(lval - rptr[d]);
                    cbuf[d] = (uchar)diff;
                    hsad[d] = (int)(hsad[d] + diff);
                }
                htext[y] += tab[lval];
            }
        }

        // initialize the left and right borders of the disparity map
        for (y = 0; y < height; y++) {
            for (x = 0; x < lofs; x++)
                dptr[y * dstep + x] = FILTERED;
            for (x = lofs + width1; x < width; x++)
                dptr[y * dstep + x] = FILTERED;
        }
        dptr += lofs;

        // std::cout << "Height is: " << height << "\nWidth is: " << width1 << std::endl; // @Alex

        for (x = 0; x < width1; x++, dptr++) {
            int *costptr = cost.data ? cost.ptr<int>() + lofs + x : &costbuf;
            int x0 = x - wsz2 - 1, x1 = x + wsz2;
            const uchar *cbuf_sub = cbuf0 + ((x0 + wsz2 + 1) % (wsz + 1)) * cstep - dy0 * ndisp;
            cbuf = cbuf0 + ((x1 + wsz2 + 1) % (wsz + 1)) * cstep - dy0 * ndisp;
            hsad = hsad0 - dy0 * ndisp;
            lptr_sub = lptr0 + MIN(MAX(x0, -lofs), width - 1 - lofs) - dy0 * sstep;
            lptr = lptr0 + MIN(MAX(x1, -lofs), width - 1 - lofs) - dy0 * sstep;
            rptr = rptr0 + MIN(MAX(x1, -rofs), width - ndisp - rofs) - dy0 * sstep;

            for (y = -dy0; y < height + dy1; y++, cbuf += ndisp, cbuf_sub += ndisp, hsad += ndisp, lptr += sstep,
                lptr_sub += sstep, rptr += sstep) {
                int lval = lptr[0];
                d = 0;

                for (; d < ndisp; d++) {
                    int diff = std::abs(lval - rptr[d]);
                    cbuf[d] = (uchar)diff;
                    hsad[d] = hsad[d] + diff - cbuf_sub[d];
                }
                htext[y] += tab[lval] - tab[lptr_sub[0]];
            }

            // fill borders
            for (y = dy1; y <= wsz2; y++)
                htext[height + y] = htext[height + dy1 - 1];
            for (y = -wsz2 - 1; y < -dy0; y++)
                htext[y] = htext[-dy0];

            // initialize sums
            for (d = 0; d < ndisp; d++)
                sad[d] = (int)(hsad0[d - ndisp * dy0] * (wsz2 + 2 - dy0));

            hsad = hsad0 + (1 - dy0) * ndisp;
            for (y = 1 - dy0; y < wsz2; y++, hsad += ndisp) {
                d = 0;
                for (; d < ndisp; d++)
                    sad[d] = (int)(sad[d] + hsad[d]);
            }
            int tsum = 0;
            for (y = -wsz2 - 1; y < wsz2; y++)
                tsum += htext[y];

            // finally, start the real processing
            for (y = 0; y < height; y++) {
                int minsad = INT_MAX, mind = -1;
                hsad = hsad0 + MIN(y + wsz2, height + dy1 - 1) * ndisp;
                hsad_sub = hsad0 + MAX(y - wsz2 - 1, -dy0) * ndisp;

                /*
                    Here the default code would calculate the min sad (cost) and min disparity for this pixel.
                    We will try to save all the characteristics of the ML algorithm.
                 */

                long long averageSad = 0;
                int second_mind = -1, third_mind = -1, fourth_mind = -1;
                int second_minsad = INT_MAX, third_minsad = INT_MAX, fourth_minsad = INT_MAX;
                double ratio_first_second = 0, ratio_first_third = 0, ratio_first_fourth, ratio_left = 0,
                       ratio_right = 0;
                int distance_first_second = 0, distance_first_third, distance_first_fourth;

                for (int d = 0; d < ndisp; d++) {
                    int currsad = sad[d] + hsad[d] - hsad_sub[d];
                    sad[d] = currsad;
                    if (currsad < minsad) {
                        // @Alex start
                        fourth_minsad = third_minsad;
                        third_minsad = second_minsad;
                        second_minsad = minsad;
                        fourth_mind = third_mind;
                        third_mind = second_mind;
                        second_mind = mind;
                        // @Alex stop

                        minsad = currsad;
                        mind = d;
                    } else if (sad[d] < second_minsad) {
                        fourth_minsad = third_minsad;
                        third_minsad = second_minsad;
                        fourth_mind = third_mind;
                        third_mind = second_mind;
                        second_minsad = sad[d];
                        second_mind = d;
                    } else if (sad[d] < third_minsad) {
                        fourth_minsad = third_minsad;
                        fourth_mind = third_mind;
                        third_minsad = sad[d];
                        third_mind = d;
                    } else if (sad[d] < fourth_minsad) {
                        fourth_minsad = sad[d];
                        fourth_mind = d;
                    }

                    // Calculate average cost @Alex
                    averageSad += sad[d];
                }
                averageSad /= ndisp;
                int dispError = ((ndisp - mind) - disparityTruth.at<uchar>(y + block_size_tmp / 2, x + ndisp));

                // DO NOT FORGET: In BM implementation, actual minimum disparity is equal to = (number of disparities -
                // minimum disparity)

                if (ml_csv.is_open() && ml_all_csv.is_open()) {
                    m_valid.at<int>(y, x + ndisp) = (ndisp - mind);
                    if (minsad != 0) { // Prevent division with 0
                        ratio_first_second = ((double)second_minsad / (double)minsad);
                        ratio_first_third = ((double)third_minsad / (double)minsad);
                        ratio_first_fourth = ((double)fourth_minsad / (double)minsad);
                        if (mind > 0 && minsad != sad[mind - 1])
                            ratio_left = ((double)sad[mind - 1] / (double)minsad);
                        else
                            ratio_left = 1;
                        if (mind < ndisp - 1 && minsad != sad[mind + 1])
                            ratio_right = ((double)sad[mind + 1] / (double)minsad);
                        else
                            ratio_right = 1;
                        distance_first_second = ((ndisp - mind) - (ndisp - second_mind));
                        distance_first_third = ((ndisp - mind) - (ndisp - third_mind));
                        distance_first_fourth = ((ndisp - mind) - (ndisp - fourth_mind));

                        if (disparityTruth.at<uchar>(y + block_size_tmp / 2, x + ndisp) != 0) {
                            // Features

                            if (dispError > 2) {
                                badPixels++;
                            }
                            goodPixels++;
                        }

                        // We in general store all the information from every pixel.

                    } else {
                        // If minimum cost is zero, then we know that disparity error is huge.
                        uncertainty_map.at<uchar>(y + block_size_tmp / 2, x + ndisp) = ndisp;
                    }
                }

                disp_unfiltered.at<int>(y + block_size_tmp / 2, x + ndisp) = (ndisp - mind);

                // NOW FILTER CHECKING WILL BE PERFORMED
                tsum += htext[y + wsz2] - htext[y - wsz2 - 1];
                if (tsum < textureThreshold) {
                    dptr[y * dstep] = FILTERED;
                    continue;
                }

                if (uniquenessRatio > 0) {
                    int thresh = minsad + (minsad * uniquenessRatio / 100);
                    for (d = 0; d < ndisp; d++) {
                        if ((d < mind - 1 || d > mind + 1) && sad[d] <= thresh)
                            break;
                    }
                    if (d < ndisp) {
                        dptr[y * dstep] = FILTERED;
                        continue;
                    }
                }

                {
                    sad[-1] = sad[1];
                    sad[ndisp] = sad[ndisp - 2];
                    int p = sad[mind + 1], n = sad[mind - 1];
                    d = p + n - 2 * sad[mind] + std::abs(p - n);
                    dptr[y * dstep] = dispDescale<mType>(ndisp - mind - 1 + mindisp, p - n, d);

                    costptr[y * coststep] = sad[mind];
                }
            }
        }
    }

    struct PrefilterInvoker : public ParallelLoopBody {
        PrefilterInvoker(const Mat &left0, const Mat &right0, Mat &left, Mat &right, uchar *buf0, uchar *buf1,
                         StereoBMParams *_state) {
            imgs0[0] = &left0;
            imgs0[1] = &right0;
            imgs[0] = &left;
            imgs[1] = &right;
            buf[0] = buf0;
            buf[1] = buf1;
            state = _state;
        }

        void operator()(const Range &range) const CV_OVERRIDE {
            for (int i = range.start; i < range.end; i++) {
                if (state->preFilterType == StereoBM::PREFILTER_NORMALIZED_RESPONSE)
                    prefilterNorm(*imgs0[i], *imgs[i], state->preFilterSize, state->preFilterCap, buf[i]);
                else
                    prefilterXSobel(*imgs0[i], *imgs[i], state->preFilterCap);
            }
        }

        const Mat *imgs0[2];
        Mat *imgs[2];
        uchar *buf[2];
        StereoBMParams *state;
    };

    struct FindStereoCorrespInvoker : public ParallelLoopBody {
        FindStereoCorrespInvoker(const Mat &_left, const Mat &_right, Mat &_disp, StereoBMParams *_state, int _nstripes,
                                 size_t _stripeBufSize, bool _useShorts, Rect _validDisparityRect, Mat &_slidingSumBuf,
                                 Mat &_cost) {
            CV_Assert(_disp.type() == CV_16S || _disp.type() == CV_32S);
            left = &_left;
            right = &_right;
            disp = &_disp;
            state = _state;
            nstripes = _nstripes;
            stripeBufSize = _stripeBufSize;
            useShorts = _useShorts;
            validDisparityRect = _validDisparityRect;
            slidingSumBuf = &_slidingSumBuf;
            cost = &_cost;
        }

        void operator()(const Range &range) const CV_OVERRIDE {
            int cols = left->cols, rows = left->rows;
            // std::cout << "Range start " << range.start << " Range ends " << range.end << std::endl;
            int _row0 = std::min(cvRound(range.start * rows / nstripes), rows);
            int _row1 = std::min(cvRound(range.end * rows / nstripes), rows);
            uchar *ptr = slidingSumBuf->ptr() + range.start * stripeBufSize;

            int dispShift = disp->type() == CV_16S ? DISPARITY_SHIFT_16S : DISPARITY_SHIFT_32S;
            int FILTERED = (state->minDisparity - 1) << dispShift;

            Rect roi = validDisparityRect & Rect(0, _row0, cols, _row1 - _row0);
            if (roi.height == 0)
                return;
            int row0 = roi.y;
            int row1 = roi.y + roi.height;

            Mat part;
            if (row0 > _row0) {
                part = disp->rowRange(_row0, row0);
                part = Scalar::all(FILTERED);
            }
            if (_row1 > row1) {
                part = disp->rowRange(row1, _row1);
                part = Scalar::all(FILTERED);
            }

            Mat left_i = left->rowRange(row0, row1);
            Mat right_i = right->rowRange(row0, row1);
            Mat disp_i = disp->rowRange(row0, row1);
            Mat cost_i = state->disp12MaxDiff >= 0 ? cost->rowRange(row0, row1) : Mat();

            // if (_row0 < 100) {
            //     sleep(1);
            // }
            // std::cout << "start row is = " << row0 << " and _row is " << _row0 << std::endl;
            // std::cout << "final row is: " << row1 << " and _ro1 is " << _row1 << std::endl;

#if CV_SIMD128
            if (useShorts) {
                if (disp_i.type() == CV_16S)
                    findStereoCorrespondenceBM_SIMD<short>(left_i, right_i, disp_i, cost_i, *state, ptr, row0,
                                                           rows - row1, _row0, row1);
                else
                    findStereoCorrespondenceBM_SIMD<int>(left_i, right_i, disp_i, cost_i, *state, ptr, row0,
                                                         rows - row1, _row0, row1);
            } else
#endif
            {
                if (disp_i.type() == CV_16S)
                    findStereoCorrespondenceBM<short>(left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1,
                                                      _row0, _row1);
                else
                    findStereoCorrespondenceBM<int>(left_i, right_i, disp_i, cost_i, *state, ptr, row0, rows - row1,
                                                    _row0, _row1);
            }

            if (state->disp12MaxDiff >= 0)
                validateDisparity(disp_i, cost_i, state->minDisparity, state->numDisparities, state->disp12MaxDiff);

            if (roi.x > 0) {
                part = disp_i.colRange(0, roi.x);
                part = Scalar::all(FILTERED);
            }
            if (roi.x + roi.width < cols) {
                part = disp_i.colRange(roi.x + roi.width, cols);
                part = Scalar::all(FILTERED);
            }
        }

        protected:
        const Mat *left, *right;
        Mat *disp, *slidingSumBuf, *cost;
        StereoBMParams *state;

        int nstripes;
        size_t stripeBufSize;
        bool useShorts;
        Rect validDisparityRect;
    };

    class StereoBMImpl CV_FINAL : public StereoBM {
        public:
        StereoBMImpl() { params = StereoBMParams(); }

        StereoBMImpl(int _numDisparities, int _SADWindowSize) {
            params = StereoBMParams(_numDisparities, _SADWindowSize);
        }

        void compute(InputArray leftarr, InputArray rightarr, OutputArray disparr) CV_OVERRIDE {

            int dtype = disparr.fixedType() ? disparr.type() : params.dispType;
            Size leftsize = leftarr.size();

            if (leftarr.size() != rightarr.size())
                CV_Error(Error::StsUnmatchedSizes, "All the images must have the same size");

            if (leftarr.type() != CV_8UC1 || rightarr.type() != CV_8UC1)
                CV_Error(Error::StsUnsupportedFormat, "Both input images must have CV_8UC1");

            if (dtype != CV_16SC1 && dtype != CV_32FC1)
                CV_Error(Error::StsUnsupportedFormat, "Disparity image must have CV_16SC1 or CV_32FC1 format");

            if (params.preFilterType != PREFILTER_NORMALIZED_RESPONSE && params.preFilterType != PREFILTER_XSOBEL)
                CV_Error(Error::StsOutOfRange, "preFilterType must be = CV_STEREO_BM_NORMALIZED_RESPONSE");

            if (params.preFilterSize < 5 || params.preFilterSize > 255 || params.preFilterSize % 2 == 0)
                CV_Error(Error::StsOutOfRange, "preFilterSize must be odd and be within 5..255");

            if (params.preFilterCap < 1 || params.preFilterCap > 63)
                CV_Error(Error::StsOutOfRange, "preFilterCap must be within 1..63");

            if (params.SADWindowSize < 5 || params.SADWindowSize > 255 || params.SADWindowSize % 2 == 0 ||
                params.SADWindowSize >= std::min(leftsize.width, leftsize.height))
                CV_Error(Error::StsOutOfRange,
                         "SADWindowSize must be odd, be within 5..255 and be not larger than image width or height");

            if (params.numDisparities <= 0 || params.numDisparities % 16 != 0)
                CV_Error(Error::StsOutOfRange, "numDisparities must be positive and divisible by 16");

            if (params.textureThreshold < 0)
                CV_Error(Error::StsOutOfRange, "texture threshold must be non-negative");

            if (params.uniquenessRatio < 0)
                CV_Error(Error::StsOutOfRange, "uniqueness ratio must be non-negative");

            int disp_shift;
            if (dtype == CV_16SC1)
                disp_shift = DISPARITY_SHIFT_16S;
            else
                disp_shift = DISPARITY_SHIFT_32S;

            int FILTERED = (params.minDisparity - 1) << disp_shift;

            Mat left0 = leftarr.getMat(), right0 = rightarr.getMat();
            disparr.create(left0.size(), dtype);
            Mat disp0 = disparr.getMat();

            preFilteredImg0.create(left0.size(), CV_8U);
            preFilteredImg1.create(left0.size(), CV_8U);
            cost.create(left0.size(), CV_16S);

            Mat left = preFilteredImg0, right = preFilteredImg1;

            int mindisp = params.minDisparity;
            int ndisp = params.numDisparities;

            int width = left0.cols;
            int height = left0.rows;
            int lofs = std::max(ndisp - 1 + mindisp, 0);
            int rofs = -std::min(ndisp - 1 + mindisp, 0);
            int width1 = width - rofs - ndisp + 1;

            if (lofs >= width || rofs >= width || width1 < 1) {
                disp0 = Scalar::all(FILTERED * (disp0.type() < CV_32F ? 1 : 1. / (1 << disp_shift)));
                return;
            }

            Mat disp = disp0;
            if (dtype == CV_32F) {
                dispbuf.create(disp0.size(), CV_32S);
                disp = dispbuf;
            }

            int wsz = params.SADWindowSize;
            int bufSize0 = (int)((ndisp + 2) * sizeof(int));
            bufSize0 += (int)((height + wsz + 2) * ndisp * sizeof(int));
            bufSize0 += (int)((height + wsz + 2) * sizeof(int));
            bufSize0 += (int)((height + wsz + 2) * ndisp * (wsz + 2) * sizeof(uchar) + 256);

            int bufSize1 = (int)((width + params.preFilterSize + 2) * sizeof(int) + 256);
            int bufSize2 = 0;
            if (params.speckleRange >= 0 && params.speckleWindowSize > 0)
                bufSize2 = width * height * (sizeof(Point_<short>) + sizeof(int) + sizeof(uchar));

            bool useShorts = params.preFilterCap <= 31 && params.SADWindowSize <= 21;
            const double SAD_overhead_coeff = 10.0;
            double N0 = 8000000 / (useShorts ? 1 : 4); // approx tbb's min number instructions reasonable for one thread
            double maxStripeSize =
                std::min(std::max(N0 / (width * ndisp), (wsz - 1) * SAD_overhead_coeff), (double)height);
            int nstripes = cvCeil(height / maxStripeSize);
            int bufSize = std::max(bufSize0 * nstripes, std::max(bufSize1 * 2, bufSize2));

            if (slidingSumBuf.cols < bufSize)
                slidingSumBuf.create(1, bufSize, CV_8U);

            uchar *_buf = slidingSumBuf.ptr();

            parallel_for_(Range(0, 2), PrefilterInvoker(left0, right0, left, right, _buf, _buf + bufSize1, &params), 1);

            Rect validDisparityRect(0, 0, width, height), R1 = params.roi1, R2 = params.roi2;
            validDisparityRect =
                getValidDisparityROI(!R1.empty() ? R1 : validDisparityRect, !R2.empty() ? R2 : validDisparityRect,
                                     params.minDisparity, params.numDisparities, params.SADWindowSize);

#ifdef SEQUENTIAL
            nstripes = 1;
#else
            nstripes = 4;
#endif
            // nstripes = 2;

            parallel_for_(Range(0, nstripes),
                          FindStereoCorrespInvoker(left, right, disp, &params, nstripes, bufSize0, useShorts,
                                                   validDisparityRect, slidingSumBuf, cost));

            if (params.speckleRange >= 0 && params.speckleWindowSize > 0)
                filterSpeckles(disp, FILTERED, params.speckleWindowSize, params.speckleRange, slidingSumBuf);

            if (disp0.data != disp.data)
                disp.convertTo(disp0, disp0.type(), 1. / (1 << disp_shift), 0);
        }

        int getMinDisparity() const CV_OVERRIDE { return params.minDisparity; }
        void setMinDisparity(int minDisparity) CV_OVERRIDE { params.minDisparity = minDisparity; }

        int getNumDisparities() const CV_OVERRIDE { return params.numDisparities; }
        void setNumDisparities(int numDisparities) CV_OVERRIDE { params.numDisparities = numDisparities; }

        int getBlockSize() const CV_OVERRIDE { return params.SADWindowSize; }
        void setBlockSize(int blockSize) CV_OVERRIDE { params.SADWindowSize = blockSize; }

        int getSpeckleWindowSize() const CV_OVERRIDE { return params.speckleWindowSize; }
        void setSpeckleWindowSize(int speckleWindowSize) CV_OVERRIDE { params.speckleWindowSize = speckleWindowSize; }

        int getSpeckleRange() const CV_OVERRIDE { return params.speckleRange; }
        void setSpeckleRange(int speckleRange) CV_OVERRIDE { params.speckleRange = speckleRange; }

        int getDisp12MaxDiff() const CV_OVERRIDE { return params.disp12MaxDiff; }
        void setDisp12MaxDiff(int disp12MaxDiff) CV_OVERRIDE { params.disp12MaxDiff = disp12MaxDiff; }

        int getPreFilterType() const CV_OVERRIDE { return params.preFilterType; }
        void setPreFilterType(int preFilterType) CV_OVERRIDE { params.preFilterType = preFilterType; }

        int getPreFilterSize() const CV_OVERRIDE { return params.preFilterSize; }
        void setPreFilterSize(int preFilterSize) CV_OVERRIDE { params.preFilterSize = preFilterSize; }

        int getPreFilterCap() const CV_OVERRIDE { return params.preFilterCap; }
        void setPreFilterCap(int preFilterCap) CV_OVERRIDE { params.preFilterCap = preFilterCap; }

        int getTextureThreshold() const CV_OVERRIDE { return params.textureThreshold; }
        void setTextureThreshold(int textureThreshold) CV_OVERRIDE { params.textureThreshold = textureThreshold; }

        int getUniquenessRatio() const CV_OVERRIDE { return params.uniquenessRatio; }
        void setUniquenessRatio(int uniquenessRatio) CV_OVERRIDE { params.uniquenessRatio = uniquenessRatio; }

        int getSmallerBlockSize() const CV_OVERRIDE { return 0; }
        void setSmallerBlockSize(int) CV_OVERRIDE {}

        Rect getROI1() const CV_OVERRIDE { return params.roi1; }
        void setROI1(Rect roi1) CV_OVERRIDE { params.roi1 = roi1; }

        Rect getROI2() const CV_OVERRIDE { return params.roi2; }
        void setROI2(Rect roi2) CV_OVERRIDE { params.roi2 = roi2; }

        void write(FileStorage &fs) const CV_OVERRIDE {
            writeFormat(fs);
            fs << "name" << name_ << "minDisparity" << params.minDisparity << "numDisparities" << params.numDisparities
               << "blockSize" << params.SADWindowSize << "speckleWindowSize" << params.speckleWindowSize
               << "speckleRange" << params.speckleRange << "disp12MaxDiff" << params.disp12MaxDiff << "preFilterType"
               << params.preFilterType << "preFilterSize" << params.preFilterSize << "preFilterCap"
               << params.preFilterCap << "textureThreshold" << params.textureThreshold << "uniquenessRatio"
               << params.uniquenessRatio;
        }

        void read(const FileNode &fn) CV_OVERRIDE {
            FileNode n = fn["name"];
            CV_Assert(n.isString() && String(n) == name_);
            params.minDisparity = (int)fn["minDisparity"];
            params.numDisparities = (int)fn["numDisparities"];
            params.SADWindowSize = (int)fn["blockSize"];
            params.speckleWindowSize = (int)fn["speckleWindowSize"];
            params.speckleRange = (int)fn["speckleRange"];
            params.disp12MaxDiff = (int)fn["disp12MaxDiff"];
            params.preFilterType = (int)fn["preFilterType"];
            params.preFilterSize = (int)fn["preFilterSize"];
            params.preFilterCap = (int)fn["preFilterCap"];
            params.textureThreshold = (int)fn["textureThreshold"];
            params.uniquenessRatio = (int)fn["uniquenessRatio"];
            params.roi1 = params.roi2 = Rect();
        }

        StereoBMParams params;
        Mat preFilteredImg0, preFilteredImg1, cost, dispbuf;
        Mat slidingSumBuf;

        static const char *name_;
    };

    const char *StereoBMImpl::name_ = "StereoMatcher.BM";

    /*
    Ptr<StereoBM> StereoBM::create(int _numDisparities, int _SADWindowSize)
    {
        return makePtr<StereoBMImpl>(_numDisparities, _SADWindowSize);
    }
    */
} // namespace cvAlex

using namespace cvAlex;

void openFiles(int numberOfDisparities) {
    ml_csv.open("ML/ml.csv", std::ofstream::out | std::ofstream::trunc);
    if (ml_csv.is_open()) {
        ml_csv
            << "min_cost, avg_cost, ratio_first_second, distance_first_second, ratio_first_third, distance_first_third,"
            << " texture, mind, ratio_left, ratio_right, Disparity Error\n";
    } else {
        std::cout << "problem with file";
    }
    ml_all_csv.open("ML/ml_all.csv");
    if (ml_all_csv.is_open()) {
        ml_all_csv
            << "min_cost, avg_cost, ratio_first_second, distance_first_second, ratio_first_third, distance_first_third,"
            << " texture, mind, ratio_left, ratio_right, Disparity Error\n";
    } else {
        std::cout << "problem with file";
    }
#ifdef SGBM_TRUTH
    sgbm_truth_csv.open("ML/sgbm_truth.csv", std::ofstream::out | std::ofstream::trunc);
    if (ml_csv.is_open()) {
        sgbm_truth_csv
            << "min_cost, avg_cost, ratio_first_second, distance_first_second, ratio_first_third, distance_first_third,"
            << " texture, mind, ratio_left, ratio_right, Disparity Error\n";
    } else {
        std::cout << "problem with file";
    }
#endif
    predictions_csv.open("predictions/y_predict.csv");
    if (!predictions_csv.is_open()) {
        std::cout << "problem with predictions" << std::endl;
    }
    predictions_all_csv.open("predictions/y_predict_all.csv");
    if (!predictions_all_csv.is_open()) {
        std::cout << "problem with predictions - all" << std::endl;
    }
}

void closeFiles() {
    ml_csv.close();
    ml_all_csv.close();
    predictions_all_csv.close();
}

int ml_window_size = 9;
int ml_total_win_pixels = (pow((2 * ml_window_size + 1), 2) - 1); // total number window pixels minus the center one

void uncertainty_map_calculation(Mat &img1, int numberOfDisparities) {
    int width = img1.cols, height = img1.rows;
    int kdata[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ml_total_win_pixels, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    Mat kernel(ml_window_size * 2 + 1, ml_window_size * 2 + 1, CV_32SC1, kdata);
    Mat res;
    filter2D(disp_unfiltered, res, CV_32SC1, kernel);

    for (int c = 0; c < img1.cols; ++c) {
        for (int r = 0; r < img1.rows; ++r) {
            if (m_valid.at<int>(r, c) != 0) {

                // I cannot use them since openCV includes the refinement procedure of the curve.
                int window_average_disp_deviation = 0, max_disp_dev = 0;
                for (int i = -ml_window_size; i < ml_window_size + 1; ++i) {
                    for (int j = -ml_window_size; j < ml_window_size + 1; ++j) {
                        if (c > numberOfDisparities + ml_window_size && c < width - ml_window_size &&
                            r > ml_window_size && r < height - ml_window_size) {
                            max_disp_dev = std::max(max_disp_dev, abs(disp_unfiltered.at<int>(r, c) -
                                                                      disp_unfiltered.at<int>(r + i, c + j)));
                            if (!(i == 0 && j == 0))
                                window_average_disp_deviation +=
                                    abs(disp_unfiltered.at<int>(r, c) - disp_unfiltered.at<int>(r + i, c + j));
                        }
                    }
                }

                window_average_disp_deviation /= ml_total_win_pixels;
                // window_average_disp_deviation = res.at<int>(r, c)/ml_total_win_pixels;

                if (ml_csv.is_open() && m_disp_error.at<int>(r, c) != 255) {
                    ml_csv << m_minsad.at<int>(r, c) << "," << m_average_sad.at<int>(r, c) << ","
                           << m_ratio_first_second.at<float>(r, c) << "," << m_ratio_first_third.at<float>(r, c) << ","
                           << m_ratio_first_fourth.at<float>(r, c) << "," << m_ratio_left.at<float>(r, c) << ","
                           << m_ratio_right.at<float>(r, c) << "," << abs(m_distance_first_second.at<int>(r, c)) << ","
                           << abs(m_distance_first_third.at<int>(r, c)) << ","
                           << abs(m_distance_first_fourth.at<int>(r, c)) << "," << abs(window_average_disp_deviation)
                           << "," << max_disp_dev << "," << (int)disp8_left_right_diff.at<uchar>(r, c) << ","
                           << abs(m_disp_error.at<int>(r, c)) << std::endl;
#ifdef SGBM_TRUTH
                    sgbm_truth_csv << abs(m_sgbm_error.at<int>(r, c)) << std::endl;
#endif
                }

                if (ml_all_csv.is_open()) {

                    ml_all_csv << r << "," << c << "," << m_minsad.at<int>(r, c) << "," << m_average_sad.at<int>(r, c)
                               << "," << m_ratio_first_second.at<float>(r, c) << ","
                               << m_ratio_first_third.at<float>(r, c) << "," << m_ratio_first_fourth.at<float>(r, c)
                               << "," << m_ratio_left.at<float>(r, c) << "," << m_ratio_right.at<float>(r, c) << ","
                               << abs(m_distance_first_second.at<int>(r, c)) << ","
                               << abs(m_distance_first_third.at<int>(r, c)) << ","
                               << abs(m_distance_first_fourth.at<int>(r, c)) << ","
                               << abs(window_average_disp_deviation) << "," << max_disp_dev << ","
                               << (int)disp8_left_right_diff.at<uchar>(r, c) << std::endl;
                }

                // Decision tree
                if (window_average_disp_deviation <= 18.5) {
                    if (window_average_disp_deviation <= 12.5) {
                        if (max_disp_dev <= 0.5) {
                            if (m_ratio_first_fourth.at<uchar>(r, c) <= 1.186) {
                                if (m_distance_first_second.at<uchar>(r, c) <= 19.5) {
                                    if (disp8_left_right_diff.at<uchar>(r, c) <= 1.5) {
                                        if (m_ratio_right.at<uchar>(r, c) <= 1) {
                                            uncertainty_map.at<uchar>(r, c) = 33;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 8;
                                        }
                                    } else {
                                        uncertainty_map.at<uchar>(r, c) = 15;
                                    }
                                } else {
                                    uncertainty_map.at<uchar>(r, c) = 30;
                                }
                            } else {
                                if (m_ratio_right.at<uchar>(r, c) <= 1.001) {
                                    if (m_ratio_first_second.at<uchar>(r, c) <= 1.001) {
                                        if (m_ratio_left.at<uchar>(r, c) <= 1.151) {
                                            uncertainty_map.at<uchar>(r, c) = 28;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 1.5;
                                        }
                                    } else {
                                        uncertainty_map.at<uchar>(r, c) = 49;
                                    }
                                } else {
                                    if (m_distance_first_second.at<uchar>(r, c) <= 3.5) {
                                        if (m_average_sad.at<uchar>(r, c) <= 485.5) {
                                            uncertainty_map.at<uchar>(r, c) = 12;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 2;
                                        }
                                    } else {
                                        if (m_ratio_first_second.at<uchar>(r, c) <= 1.349) {
                                            uncertainty_map.at<uchar>(r, c) = 10;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) =
                                                1; // Seems kind of innacurate, take care. Only a very few samples.
                                        }
                                    }
                                }
                            }
                        } else {
                            if (window_average_disp_deviation <= 8.5) {
                                if (disp8_left_right_diff.at<uchar>(r, c) <= 1.5) {
                                    if (window_average_disp_deviation <= 3.5) {
                                        uncertainty_map.at<uchar>(r, c) = 1;
                                    } else {
                                        if (max_disp_dev <= 30.5) {
                                            uncertainty_map.at<uchar>(r, c) = 3; // maybe also 3 but too few samples tbh
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 1;
                                        }
                                    }
                                } else {
                                    uncertainty_map.at<uchar>(r, c) = 30;
                                    if (window_average_disp_deviation <= 5.5) {
                                        if (m_ratio_first_fourth.at<uchar>(r, c) <= 1.002) {
                                            uncertainty_map.at<uchar>(r, c) = 10;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 1.5;
                                        }
                                    } else {
                                        if (max_disp_dev <= 31) {
                                            uncertainty_map.at<uchar>(r, c) = 7;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 3;
                                        }
                                    }
                                }
                            } else {
                                if (disp8_left_right_diff.at<uchar>(r, c) <= 1.5) {
                                    if (max_disp_dev <= 34.5) {
                                        if (window_average_disp_deviation <= 10.5) {
                                            uncertainty_map.at<uchar>(r, c) = 6;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 10;
                                        }
                                    } else {
                                        if (m_distance_first_fourth.at<uchar>(r, c) <= 16.5) {
                                            uncertainty_map.at<uchar>(r, c) = 3;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 2;
                                        }
                                    }
                                } else {
                                    uncertainty_map.at<uchar>(r, c) = 30;
                                    if (max_disp_dev <= 35.5) {
                                        uncertainty_map.at<uchar>(r, c) = 11;
                                    } else {
                                        if (m_distance_first_third.at<uchar>(r, c) <= 18.5) {
                                            uncertainty_map.at<uchar>(r, c) = 7;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 5;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (disp8_left_right_diff.at<uchar>(r, c) <= 1.5) {
                            if (window_average_disp_deviation <= 15.5) {
                                if (max_disp_dev <= 38.5) {
                                    uncertainty_map.at<uchar>(r, c) = 10;
                                } else {
                                    if (m_distance_first_third.at<uchar>(r, c) <= 26.5) {
                                        if (m_ratio_first_third.at<uchar>(r, c) <= 1.231) {
                                            uncertainty_map.at<uchar>(r, c) = 7;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 4;
                                        }
                                    } else {
                                        if (max_disp_dev <= 47.5) {
                                            uncertainty_map.at<uchar>(r, c) = 5;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 3;
                                        }
                                    }
                                }
                            } else {
                                uncertainty_map.at<uchar>(r, c) = 30;
                                if (max_disp_dev <= 48.5) {
                                    uncertainty_map.at<uchar>(r, c) = 14;
                                } else {
                                    if (m_distance_first_fourth.at<uchar>(r, c) <= 30.5) {
                                        uncertainty_map.at<uchar>(r, c) = 9;
                                    } else {
                                        if (m_distance_first_second.at<uchar>(r, c) <= 2.5) {
                                            uncertainty_map.at<uchar>(r, c) = 4;
                                        } else {
                                            uncertainty_map.at<uchar>(r, c) = 7;
                                        }
                                    }
                                }
                            }
                        } else {
                            uncertainty_map.at<uchar>(r, c) = 15;
                        }
                    }
                } else {
                    uncertainty_map.at<uchar>(r, c) = 70;
                }
            }
        }
    }

    // GaussianBlur( uncertainty_map, uncertainty_map, Size( 9, 9), 0, 0 );
}

void RemoveOutliersWithPredictor(Mat uncertainty_map, Mat &disp8) {
    for (int i = 0; i < uncertainty_map.rows; i++) {
        for (int j = 0; j < uncertainty_map.cols; j++) {
            uchar value = uncertainty_map.at<uchar>(i, j);
            if (value > 3) {
                disp8.at<uchar>(i, j) = 0;
            }
        }
    }
}

int main(int argc, char **argv) {

    // Read the left, right images and the disparity truth
    int color_mode = cv::IMREAD_GRAYSCALE;
    Mat img1 = imread(argv[1], color_mode);
    Mat img2 = imread(argv[2], color_mode);
    disparityTruth = imread(argv[3], color_mode);
    img1.convertTo(img1, CV_8UC1);
    img2.convertTo(img2, CV_8UC1);
    disparityTruth.convertTo(disparityTruth, CV_8UC1);

#ifdef SGBM_TRUTH
    sgbmTruth = imread(argv[4], color_mode);
    sgbmTruth.convertTo(sgbmTruth, CV_8UC1);
#endif

#ifdef DOWNSAMPLE
    cv::resize(img1, img1, cv::Size(), 0.75, 0.75, INTER_NEAREST);
    cv::resize(img2, img2, cv::Size(), 0.75, 0.75, INTER_NEAREST);
    cv::resize(disparityTruth, disparityTruth, cv::Size(), 0.75, 0.75, INTER_NEAREST);
#endif

    // Check if everything is okay
    if (img1.empty()) {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }
    if (img2.empty()) {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }
    if (disparityTruth.empty()) {
        printf("Command-line parameter error: could not load the disparity truth image file\n");
        return -1;
    }

    // Reserve space for the vectors
    Size img_size = img1.size();
    long long total_pixels = img_size.height * img_size.width;

#ifdef PRINT_TO_FILES
    openFiles(numberOfDisparities);
#endif

    // BM calculations
    Ptr<StereoBMImpl> bm = makePtr<cvAlex::StereoBMImpl>(16, block_size_tmp);
    Ptr<StereoBM> bm_right = StereoBM::create(16, block_size_tmp);

    // BM initializations

    Rect roi1, roi2;
    Mat Q;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterType(StereoBM::PREFILTER_XSOBEL);
    bm->setPreFilterCap(31); // The original was with normalization filter and cap at 31.
    bm->setBlockSize(block_size_tmp);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);

    bm_right->setROI1(roi1);
    bm_right->setROI2(roi2);
    bm_right->setPreFilterType(StereoBM::PREFILTER_XSOBEL);
    bm_right->setPreFilterCap(31); // The original was with normalization filter and cap at 31.
    bm_right->setBlockSize(block_size_tmp);
    bm_right->setMinDisparity(0);
    bm_right->setNumDisparities(numberOfDisparities);

    /*
    To make sure that there is enough texture to overcome random noise during match‐
    ing, OpenCV also employs a texture threshold. This is just a limit on the SAD win‐
    dow response such that no match is considered whose response is below some
    minimal value.
     */
    int textureThreshold = 0;
    bm->setTextureThreshold(textureThreshold);
    int uniquenessRatio = 0;
    bm->setUniquenessRatio(uniquenessRatio); // original was set to 10
    bm->setSpeckleWindowSize(-1);            // 100
    bm->setSpeckleRange(-1);                 // 32
    bm->setDisp12MaxDiff(255);

    bm_right->setTextureThreshold(0);
    bm_right->setUniquenessRatio(0); // original was set to 10
    bm_right->setSpeckleWindowSize(-1);
    bm_right->setSpeckleRange(-1);
    bm_right->setDisp12MaxDiff(255);

    // Compute left disparity image
    Mat disp, disp8;
    bm->compute(img1, img2, disp);
    disp.convertTo(disp8, CV_8U, 255 / (255 * 16.));

    // Compute right disparity image
    Mat disp_right(height_tmp, width_tmp, CV_16SC1, Scalar(0));
    Mat disp8_right(height_tmp, width_tmp, CV_8UC1, Scalar(0));
    Mat right_img = img2.clone(), left_img = img1.clone();
    // flip horizontally
    flip(right_img, right_img, 1);
    flip(left_img, left_img, 1);
    bm_right->compute(right_img, left_img, disp_right);
    // flip horizontally the disparity map
    flip(disp_right, disp_right, 1);
    disp_right.convertTo(disp8_right, CV_8U, 1 / (16.0));

    disp8_left_right_diff = disp8.clone();
    for (int i = 0; i < disp8.rows; i++) {
        for (int j = numberOfDisparities; j < disp8.cols; j++) {
            uchar d = disp8.at<uchar>(i, j);
            // first check if we are into right image boundaries
            if (j - d < 0) {
                disp8_left_right_diff.at<uchar>(i, j) = 255;
            } else {
                int disp_diff = abs(static_cast<int>(disp8_right.at<uchar>(i, j - d)) - d);
                disp8_left_right_diff.at<uchar>(i, j) = static_cast<uchar>(disp_diff);
            }
        }
    }

    // SHOW THE RESULTS
    cv::convertScaleAbs(disp8, disp8);

#ifdef WRITE_IMAGE
    imwrite("disp8.png", disp8);
    // imshow("right disparity", disp8_right);
    imwrite("disp_right.png", disp8_right);
    // imshow("disparity difference between left and right", disp8_left_right_diff);
    imwrite("disp_diff.png", disp8_left_right_diff);
    std::string img_name = argv[1];
    img_name = img_name.substr(img_name.length() - 13);
    imwrite("images/disparity_map/" + img_name, disp8);
#endif
    // Save disparity image in the disparity_map folder.

    uncertainty_map_calculation(disp8, numberOfDisparities);
#ifdef PRINT_TO_FILES
    closeFiles();
#endif

    Mat disp8_uncertainty = disp8.clone();
    RemoveOutliersWithPredictor(uncertainty_map, disp8_uncertainty);
    // RemoveOutliersWithPredictor2(uncertainty_map, disp8_uncertainty);

    // imshow("uncertainty map", uncertainty_map);
#ifdef WRITE_IMAGE
    imwrite("uncertainty_map.png", uncertainty_map);
    imwrite("certain_disparity_map.png", disp8_uncertainty);
#endif
    // equalizeHist(disp8_uncertainty, disp8_uncertainty);
    // imshow("left image", img1);
    // imshow("certain disparity map", disp8_uncertainty);
    // waitKey();
    double badPercentage = (badPixels * 1.0f * 100 / goodPixels);
    std::cout << "Bad percentage: " << badPercentage << std::endl;
    std::cout << "Average disp Error = " << avg_dispError / goodPixels << std::endl;
    std::cout << "\n\n";

#ifdef ORIGINAL_STEREO
    Ptr<StereoBMImpl> bm_filtered = makePtr<cvAlex::StereoBMImpl>(16, block_size_tmp);
    bm_filtered->setPreFilterType(StereoBM::PREFILTER_XSOBEL);
    bm_filtered->setPreFilterCap(31); // The original was with normalization filter and cap at 31.
    bm_filtered->setBlockSize(block_size_tmp);
    bm_filtered->setMinDisparity(0);
    bm_filtered->setNumDisparities(numberOfDisparities);
    textureThreshold = 300;
    bm_filtered->setTextureThreshold(textureThreshold);
    uniquenessRatio = 10;
    bm_filtered->setUniquenessRatio(uniquenessRatio); // original was set to 10
    bm_filtered->setSpeckleWindowSize(100);           // 100
    bm_filtered->setSpeckleRange(32);                 // 32
    bm_filtered->setDisp12MaxDiff(1);
    Mat disp_filtered, disp8_filtered;
    bm_filtered->compute(img1, img2, disp_filtered);
    disp_filtered.convertTo(disp8_filtered, CV_8U, 255 / (255 * 16.));
    // imwrite("disp8_filtered.png", disp8_filtered);
    // equalizeHist(disp8_filtered, disp8_filtered);
    // imshow("filtered disparity map", disp8_filtered);
    // waitKey();
#endif
    cv::convertScaleAbs(m_disp_error2, m_disp_error2);

    int start_x = 90, start_y = 15;
    cv::Rect myROI(start_x, start_y, img1.cols - 20 - start_x, img1.rows - 30 - start_y);

    Mat img_disp_error = m_disp_error2(myROI);
    cv::equalizeHist(img_disp_error, img_disp_error);
    applyColorMap(img_disp_error, img_disp_error, COLORMAP_JET);

    disparityTruth = disparityTruth(myROI);
    for (int i = 0; i < img_disp_error.rows; i++) {
        for (int j = 0; j < img_disp_error.cols; j++) {
            if (disparityTruth.at<uchar>(i, j) == 0) {
                img_disp_error.at<Vec3b>(i, j) = 0;
            }
        }
    }

    imwrite("disp_error.png", img_disp_error);

    Mat disparityTruthColored = disparityTruth;
    cv::equalizeHist(disparityTruthColored, disparityTruthColored);
    applyColorMap(disparityTruthColored, disparityTruthColored, COLORMAP_JET);
    for (int i = 0; i < img_disp_error.rows; i++) {
        for (int j = 0; j < img_disp_error.cols; j++) {
            if (disparityTruth.at<uchar>(i, j) == 0) {
                disparityTruthColored.at<Vec3b>(i, j) = 0;
            }
        }
    }
    imwrite("disp_truth_colored.png", disparityTruthColored);

    return 0;
}