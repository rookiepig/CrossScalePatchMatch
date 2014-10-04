///////////////////////////////////////////////////////
// File: cspc.h
// Desc: corss scale cost aggregation
// 
// Author: rookiepig
// Date: 2014/04/03
//
///////////////////////////////////////////////////////
#pragma once
#include"../commfunc.h"
#include"i_plane_cost.h"

#define COST_ALPHA 0.1
#define TAU_CLR 10.0
#define TAU_GRD 2.0
// 10 * 3 = 30 means divide color by 3
#define WGT_GAMMA  10.0
//#define REG_LAMBDA 1

class CSPC : public IPlaneCost {
 public:
   CSPC(const Mat& l_img, const Mat& r_img,
     const int& max_disp, const int& wnd_size,
     const int& scale_num, const double& reg_lambda);
     //const double& alpha,
     //const double& tau_clr, const double& tau_grd,
     //const double& gamma);
   ~CSPC(void);
   virtual double GetPlaneCost(
     const int& ref_x,
     const int& ref_y,
     const Plane& plane,
     const RefView& view
    ) const;
 private:
   double GetCostWeight(const int& ref_x,
     const int& ref_y, const int& q_x, const int& q_y,
     const RefView& view, const int& scale) const;
   double GetPixelCost(const int& ref_x, const int& ref_y,
     const double& other_x, const int& other_y,
     const RefView& view, const int& scale) const;
   // multi-scale related
   int scale_num_;
   double* scale_wgt_;
   // color image
   Mat* img_[kViewNum];
   Mat* lab_[kViewNum];
   // gradient along x axis
   Mat* grd_x_[kViewNum];
   // image property
   int* wid_;
   int* hei_;
   // look up table for fast-exp
   double* lookup_exp_;
   // method paramter
   int* max_disp_;
   int wnd_size_;
   int half_wnd_;
   //double alpha_;    // balance color and gradient cost
   //double tau_clr_;  // threshold for color cost
   //double tau_grd_;  // threshold for gradient cost
   //double gamma_;    // cost weight parameter
};