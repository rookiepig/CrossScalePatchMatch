///////////////////////////////////////////////////////
// File: grd_pc.cc
// Desc: implement gradient + color plane cost
// 
// Author: rookiepig
// Date: 2014/04/03
//
///////////////////////////////////////////////////////
#include"grd_pc.h"

GrdPC::GrdPC(const Mat& l_img, const Mat& r_img,
  const int& max_disp,
  const int& wnd_size, const double& alpha,
  const double& tau_clr, const double& tau_grd,
  const double& gamma) :
  max_disp_(max_disp), wnd_size_(wnd_size), alpha_(alpha),
  tau_clr_(tau_clr), tau_grd_(tau_grd),
  gamma_(gamma) {
  // for TAD + Grd input image must be CV_64FC3
  CV_Assert(l_img.type() == CV_64FC3 && r_img.type() == CV_64FC3);
  cout << "\t GRD plane cost\n";
  img_[kLeft]  = l_img.clone();
  img_[kRight] = r_img.clone();
  hei_ = l_img.rows;
  wid_ = l_img.cols;
  // get x-axis gradient
  for (int v = 0; v < kViewNum; ++v) {
    Mat tmp;
    Mat gray;
    img_[v].convertTo(tmp, CV_32F);
    cvtColor(tmp, gray, CV_RGB2GRAY);
    // X Gradient
    // sobel size must be 1
    Sobel(gray, grd_x_[v], CV_64F, 1, 0, 1);
    grd_x_[v] += 0.5;
  }
  // init exp look-up table
  lookup_exp_ = new double[1000];
  for (int i = 0; i < 1000; ++i) {
    lookup_exp_[i] = exp(- i / gamma_);
  }
}

GrdPC::~GrdPC(void) {
  // do nothing
}

double GrdPC::GetPlaneCost(const int& ref_x, const int& ref_y,
  const Plane& plane,
  const RefView& view) const {
  const int half_wnd = wnd_size_ / 2;
  double cost = 0.0;
  Vec3d plane_param = plane.param();
  for (int dy = -half_wnd; dy <= half_wnd; ++dy) {
    int q_y = HandleBorder(ref_y + dy, hei_);
    for (int dx = -half_wnd; dx <= half_wnd; ++dx) {
      int q_x = HandleBorder(ref_x + dx, wid_);
      const double wgt = GetCostWeight(ref_x, ref_y, q_x, q_y,
        view);
      double other_x = 0.0;
      double q_disp = plane_param.dot(Vec3d(q_x, q_y, 1.0));
      if (q_disp <= 0.0 || q_disp >= max_disp_) {
        // impossible disparity --> largest cost
        cost += wgt *(alpha_ * tau_clr_ +
          (1 - alpha_) * tau_grd_);
      } else {
        if (view == kLeft) {
          other_x = q_x - q_disp;
        }
        else {
          other_x = q_x + q_disp;
        }
        cost += wgt * GetPixelCost(q_x, q_y, other_x, q_y, view);
      }
    }
  }
  return cost;
}

inline double GrdPC::GetCostWeight(const int& ref_x,
  const int& ref_y, const int& q_x, const int& q_y,
  const RefView& view) const {
  // assume three channel
  const double* I_p = img_[view].ptr<double>(ref_y) + 3 * ref_x;
  const double* I_q = img_[view].ptr<double>(q_y) + 3 * q_x;
  double sum = 0;
  for (int c = 0; c < 3; ++c) {
    sum += 255 * fabs(I_p[c] - I_q[c]);
  }
  return lookup_exp_[static_cast<int>(sum)];
  // return exp(-sum / gamma_);
}

double GrdPC::GetPixelCost(const int& ref_x, 
  const int& ref_y,
  const double& other_x, const int& other_y,
  const RefView& view) const {
  int floor_x = static_cast<int>(floor(other_x));
  int ceil_x  = static_cast<int>(ceil(other_x));
  const double floor_wgt = static_cast<double>(ceil_x) - other_x;
  const double ceil_wgt = 1 - floor_wgt;
  // handle special border
  floor_x = HandleBorder(floor_x, wid_);
  ceil_x  = HandleBorder(ceil_x, wid_);
  const double* I_q = img_[view].ptr<double>(ref_y) + 3 * ref_x;
  // 1 - view --> other view
  const double* I_floor = img_[1 - view].ptr<double>(other_y) +
    3 * floor_x;
  const double* I_ceil = img_[1 - view].ptr<double>(other_y) +
    3 * ceil_x;
  double clr_cost = 0.0;
  for (int c = 0; c < 3; ++c) {
    // interpolated color difference
    clr_cost += fabs(I_q[c] -
      (floor_wgt * I_floor[c] + ceil_wgt * I_ceil[c]));
  }
  double grd_cost = 0.0;
  const double* G_q = grd_x_[view].ptr<double>(ref_y) + ref_x;
  const double* G_floor = grd_x_[1 - view].ptr<double>(other_y) + 
    floor_x;
  const double* G_ceil = grd_x_[1 - view].ptr<double>(other_y) +
    ceil_x;
  // interpolated gradient difference
  grd_cost = fabs(*G_q - (floor_wgt * (*G_floor) +
    ceil_wgt * (*G_ceil)));
  return alpha_ * min(clr_cost, tau_clr_) + 
    (1 - alpha_) * min(grd_cost, tau_grd_);
}