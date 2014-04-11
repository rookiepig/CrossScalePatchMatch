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
  const int& max_disp, const int& wnd_size ) :
  //const double& alpha,
  //const double& tau_clr, const double& tau_grd,
  //const double& gamma) :
  max_disp_(max_disp), wnd_size_(wnd_size) {
  //alpha_(alpha),
  //tau_clr_(tau_clr), tau_grd_(tau_grd),
  //gamma_(gamma) {

  // for TAD + Grd input image must be CV_64FC3
  // CV_Assert(l_img.type() == CV_64FC3 && r_img.type() == CV_64FC3);
  CV_Assert(l_img.type() == CV_8UC3 && r_img.type() == CV_8UC3);
  cout << "\t GRD plane cost\n";
  img_[kLeft]  = l_img.clone();
  img_[kRight] = r_img.clone();
  hei_ = l_img.rows;
  wid_ = l_img.cols;
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    // get LAB color image
    cvtColor(img_[v], lab_[v], CV_BGR2Lab);  
    // get x-axis gradient
    // Mat tmp;
    Mat gray;
    // img_[v].convertTo(tmp, CV_32F);
    cvtColor(img_[v], gray, CV_BGR2GRAY);
    // X Gradient
    // sobel size must be 1
    Sobel(gray, grd_x_[v], CV_16S, 1, 0, 1);
    // grd_x_[v] += 0.5;
  }
  // init exp look-up table
  lookup_exp_ = new double[1000];
  for (int i = 0; i < 1000; ++i) {
    lookup_exp_[i] = exp(-i * 1.0 / WGT_GAMMA);
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
      double q_disp = plane_param[0] * q_x +
                      plane_param[1] * q_y +
                      plane_param[2];
      if (q_disp <= 0.0 || q_disp >= max_disp_) {
        // impossible disparity --> largest cost
        cost += wgt * (COST_ALPHA * TAU_CLR +
          (1 - COST_ALPHA) * TAU_GRD);
      } else {
        //if (view == kLeft) {
        //  other_x = q_x - q_disp;
        //}
        //else {
        //  other_x = q_x + q_disp;
        //}
        const double other_x = q_x + (2 * view - 1) * q_disp;
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
  const uchar* I_p = lab_[view].ptr<uchar>(ref_y) + 3 * ref_x;
  const uchar* I_q = lab_[view].ptr<uchar>(q_y) + 3 * q_x;
  int sum = abs(I_p[0] - I_q[0]) +
            abs(I_p[1] - I_q[1]) +
            abs(I_p[2] - I_q[2]);
  return lookup_exp_[sum];
  // return exp(-sum / gamma_);
}

inline double GrdPC::GetPixelCost(const int& ref_x, 
  const int& ref_y, const double& other_x, const int& other_y,
  const RefView& view) const {
 
  int floor_x = static_cast<int>(floor(other_x));
  int ceil_x = floor_x + 1;
  const double floor_wgt = ceil_x - other_x;
  // const double ceil_wgt = 1 - floor_wgt;
  // handle special border
  floor_x = HandleBorder(floor_x, wid_);
  ceil_x  = HandleBorder(ceil_x, wid_);
  const uchar* I_q = img_[view].ptr<uchar>(ref_y) + 3 * ref_x;
  // 1 - view --> other view
  const uchar* I_ohter_y = img_[1 - view].ptr<uchar>(other_y);
  const uchar* I_floor = I_ohter_y + 3 * floor_x;
  const uchar* I_ceil = I_ohter_y + 3 * ceil_x;
  // interpolated color difference
  double clr_cost = 
    fabs(I_q[0] - I_ceil[0] + floor_wgt * (I_ceil[0] - I_floor[0])) +
    fabs(I_q[1] - I_ceil[1] + floor_wgt * (I_ceil[1] - I_floor[1])) +
    fabs(I_q[2] - I_ceil[2] + floor_wgt * (I_ceil[2] - I_floor[2]));
  // for (int c = 0; c < 3; ++c) {
  // } 
  const short G_q = *(grd_x_[view].ptr<short>(ref_y) +ref_x);
  const short* G_other_y = grd_x_[1 - view].ptr<short>(other_y);
  const short G_floor = G_other_y[floor_x];
  const short G_ceil = G_other_y[ceil_x];
  // interpolated gradient difference
  double grd_cost = 
    fabs(G_q - G_ceil + floor_wgt * (G_ceil - G_floor));

  clr_cost = clr_cost > TAU_CLR ? TAU_CLR : clr_cost;
  grd_cost = grd_cost > TAU_GRD ? TAU_GRD : grd_cost;
  return COST_ALPHA * clr_cost + (1 - COST_ALPHA) * grd_cost;
}