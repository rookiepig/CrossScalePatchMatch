///////////////////////////////////////////////////////
// File: GrdPC.h
// Desc: gradient + color plane cost
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

// #define USE_BORDER
//#define USE_INTER
#ifdef USE_INTER
#define INTER_SIZE 10.0
#endif
// #define INV_DISP_COST_SCALE 10000
// #define USE_LAB_WGT

class GrdPC : public IPlaneCost {
 public:
   GrdPC(const Mat& l_img, const Mat& r_img,
     const int& max_disp,
     const int& wnd_size);
     //const double& alpha,
     //const double& tau_clr, const double& tau_grd,
     //const double& gamma);
   ~GrdPC(void);
   virtual double GetPlaneCost(
     const int& ref_x,
     const int& ref_y,
     const Plane& plane,
     const RefView& view
    ) const;
 private:
   double GetCostWeight(const int& ref_x,
     const int& ref_y, const int& q_x, const int& q_y,
     const RefView& view) const;
   double GetPixelCost(const int& ref_x, const int& ref_y,
     const double& other_x, const int& other_y,
     const RefView& view) const;
   // color image
   Mat img_[kViewNum];
   Mat lab_[kViewNum];
   // gradient along x axis
   Mat grd_x_[kViewNum];
#ifdef USE_INTER
   Mat inter_img_[kViewNum];    // interpolated image
   Mat inter_grd_x_[kViewNum];    // interpolated gradient
   int inter_wid_;
#endif
   // image property
   int wid_;
   int hei_;
   // look up table for fast-exp
   double* lookup_exp_;
   // method paramter
   int max_disp_;
   int wnd_size_;
   int half_wnd_;
   //double alpha_;    // balance color and gradient cost
   //double tau_clr_;  // threshold for color cost
   //double tau_grd_;  // threshold for gradient cost
   //double gamma_;    // cost weight parameter
};