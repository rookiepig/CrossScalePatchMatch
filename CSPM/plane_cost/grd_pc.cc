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
  half_wnd_ = wnd_size_ / 2;
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
    Sobel(gray, grd_x_[v], CV_64F, 1, 0, 1);
    // grd_x_[v] = abs(grd_x_[v]);
  }
#ifdef _DEBUG
  // view gradient image
  Mat tmp;
  grd_x_[kLeft].convertTo(tmp, CV_8U);
  imshow("l_grd", tmp);
  grd_x_[kRight].convertTo(tmp, CV_8U);
  imshow("r_grd", tmp);
  waitKey(-1);
#endif
  // init exp look-up table
  lookup_exp_ = new double[1000];
  for (int i = 0; i < 1000; ++i) {
    lookup_exp_[i] = exp(-i * 1.0 / WGT_GAMMA);
  }
}

GrdPC::~GrdPC(void) {
  // do nothing
  delete[] lookup_exp_;
}

double GrdPC::GetPlaneCost(const int& ref_x, const int& ref_y,
  const Plane& plane, const RefView& view) const {

  double cost = 0.0;
  Vec3d plane_param = plane.param();
  const double plane_a = plane_param[0];
  const double plane_b = plane_param[1];
  const double plane_c = plane_param[2];
  const uchar* I_p = img_[view].ptr<uchar>(ref_y) + 3 * ref_x;
#ifdef USE_LAB_WGT
  const uchar* lab_p = lab_[view].ptr<uchar>(ref_y) + 3 * ref_x;
#endif
  for (int dy = -half_wnd_; dy <= half_wnd_; ++dy) {
    int q_y = HandleBorder(ref_y + dy, hei_);
    const uchar* I_q_y     = img_[view].ptr<uchar>(q_y);
#ifdef USE_LAB_WGT
    const uchar* lab_q_y = lab_[view].ptr<uchar>(q_y);
#endif
    const uchar* I_ohter_y = img_[1 - view].ptr<uchar>(q_y);
    const double* G_q_y     = grd_x_[view].ptr<double>(q_y);
    const double* G_other_y = grd_x_[1 - view].ptr<double>(q_y);
    const double q_disp_y = plane_b * q_y + plane_c;
    for (int dx = -half_wnd_; dx <= half_wnd_; ++dx) {
      int q_x = HandleBorder(ref_x + dx, wid_);
      // const double wgt = GetCostWeight(ref_x, ref_y, q_x, q_y,
      //   view);
      // assume three channel
#ifdef USE_LAB_WGT
      const uchar* lab_q = lab_q_y + 3 * q_x;
      int sum = abs(lab_p[0] - lab_q[0]) +
                abs(lab_p[1] - lab_q[1]) +
                abs(lab_p[2] - lab_q[2]);
#else
      const uchar* I_q = I_q_y + 3 * q_x;
      int sum = abs(I_p[0] - I_q[0]) +
                abs(I_p[1] - I_q[1]) +
                abs(I_p[2] - I_q[2]);
#endif
      const double wgt = lookup_exp_[sum];
      double q_disp = plane_a * q_x + q_disp_y;
      int q_disp_floor = Floor2Int(q_disp);
      if (q_disp_floor <= 0 || q_disp_floor >= max_disp_) {
        // impossible disparity --> very large cost
        cost += wgt * (COST_ALPHA * TAU_CLR +
          (1 - COST_ALPHA) * TAU_GRD);
      } else {
        const double other_x = q_x + (2 * view - 1) * q_disp;
        // cost += wgt * GetPixelCost(q_x, q_y, other_x, q_y, view);
       //if (other_x > numeric_limits<int>::max()) {
       //   cout << "error: int overflow, other_x is " << other_x << endl;
       // }
        int floor_x = Floor2Int(other_x);
        int ceil_x = floor_x + 1;
        const double floor_wgt = ceil_x - other_x;
        // handle special border
        floor_x = HandleBorder(floor_x, wid_);
        ceil_x  = HandleBorder(ceil_x, wid_);
        // 1 - view --> other view
        const uchar* I_floor = I_ohter_y + 3 * floor_x;
        const uchar* I_ceil = I_ohter_y + 3 * ceil_x;
        // interpolated color difference
        double clr_cost = 
          fabs(I_q[0] - I_ceil[0] + floor_wgt * (I_ceil[0] - I_floor[0])) +
          fabs(I_q[1] - I_ceil[1] + floor_wgt * (I_ceil[1] - I_floor[1])) +
          fabs(I_q[2] - I_ceil[2] + floor_wgt * (I_ceil[2] - I_floor[2]));
        clr_cost *= 0.33333333333333;
        const double G_floor = G_other_y[floor_x];
        const double G_ceil  = G_other_y[ceil_x];
        // interpolated gradient difference
        double grd_cost = 
          fabs(G_q_y[q_x] - G_ceil + floor_wgt * (G_ceil - G_floor));

        // clr_cost = clr_cost > TAU_CLR ? TAU_CLR : clr_cost;
        if (clr_cost > TAU_CLR) {
          clr_cost = TAU_CLR;
        }
        // grd_cost = grd_cost > TAU_GRD ? TAU_GRD : grd_cost;
        if (grd_cost > TAU_GRD) {
          grd_cost = TAU_GRD;
        }
        cost += wgt * (
          COST_ALPHA * clr_cost + (1 - COST_ALPHA) * grd_cost );
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
 
  int floor_x = Floor2Int(other_x);
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
  const double G_q = *(grd_x_[view].ptr<double>(ref_y) +ref_x);
  const double* G_other_y = grd_x_[1 - view].ptr<double>(other_y);
  const double G_floor = G_other_y[floor_x];
  const double G_ceil = G_other_y[ceil_x];
  // interpolated gradient difference
  double grd_cost = 
    fabs(G_q - G_ceil + floor_wgt * (G_ceil - G_floor));

  clr_cost = clr_cost > TAU_CLR ? TAU_CLR : clr_cost;
  grd_cost = grd_cost > TAU_GRD ? TAU_GRD : grd_cost;
  return COST_ALPHA * clr_cost + (1 - COST_ALPHA) * grd_cost;
}