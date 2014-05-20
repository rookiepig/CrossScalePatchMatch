///////////////////////////////////////////////////////
// File: pre_ss_pc.cc
// Desc: precompute cost volume
//       for single-scale plane cost
//
// Author: rookiepig
// Date: 2014/05/19
//
///////////////////////////////////////////////////////
#include"pre_ss_pc.h"

PreSSPC::PreSSPC(const Mat& l_img, const Mat& r_img,
  const int& max_disp, const int& wnd_size, CCMethod* cc_method ) :
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
  // cout << "\t GRD plane cost\n";
  img_[kLeft]  = l_img.clone();
  img_[kRight] = r_img.clone();
  hei_ = l_img.rows;
  wid_ = l_img.cols;
  half_wnd_ = wnd_size_ / 2;
  // convert left and right image for cost volume
  Mat tmp_l, tmp_r;
  cvtColor(l_img, tmp_l, CV_BGR2RGB);
  cvtColor(r_img, tmp_r, CV_BGR2RGB);
  tmp_l.convertTo(tmp_l, CV_64F);
  tmp_r.convertTo(tmp_r, CV_64F);

  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    // init cost volum data
    cost_vol_[v] = new Mat[max_disp_ + 1];
    for (int d = 0; d < max_disp_ + 1; ++d) {
      cost_vol_[v][d] = Mat::zeros(hei_, wid_, CV_64FC1);
    }
    // build cost volume
    if (v == kLeft) {
      cc_method->buildCV(tmp_l, tmp_r, max_disp_ + 1, cost_vol_[v]);
    } else if (v == kRight) {
      cc_method->buildRightCV(tmp_l, tmp_r, max_disp_ + 1, cost_vol_[v]);
    }
    // record maximum cost value
    max_cost_[v] = -1.0;
    for (int d = 0; d < max_disp_ + 1; ++d) {
      double tmp = 0.0;
      minMaxLoc(cost_vol_[v][d], NULL, &tmp);
      if (tmp > max_cost_[v]) {
        max_cost_[v] = tmp;
      }
    }
  } // end for each view
  // init exp look-up table
  lookup_exp_ = new double[1000];
  for (int i = 0; i < 1000; ++i) {
    lookup_exp_[i] = exp(-i * 1.0 / WGT_GAMMA);
  }
}

PreSSPC::~PreSSPC(void) {
  delete[] lookup_exp_;
  for(int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    delete[] cost_vol_[v];
  }
}

double PreSSPC::GetPlaneCost(const int& ref_x, const int& ref_y,
  const Plane& plane, const RefView& view) const {

  double cost = 0.0;
  Vec3d plane_param = plane.param();
  const double plane_a = plane_param[0];
  const double plane_b = plane_param[1];
  const double plane_c = plane_param[2];
  const uchar* I_p = img_[view].ptr<uchar>(ref_y) + 3 * ref_x;

  for (int dy = -half_wnd_; dy <= half_wnd_; ++dy) {
    int q_y = ref_y + dy;
    if (q_y >= 0 && q_y < hei_) {
      const uchar* I_q_y = img_[view].ptr<uchar>(q_y);
      const double q_disp_y = plane_b * q_y + plane_c;
      for (int dx = -half_wnd_; dx <= half_wnd_; ++dx) {
        int q_x = ref_x + dx;
        if (q_x >= 0 && q_x < wid_) {
          // assume 3-channel!
          const uchar* I_q = I_q_y + 3 * q_x;
          int sum = abs(I_p[0] - I_q[0]) +
            abs(I_p[1] - I_q[1]) +
            abs(I_p[2] - I_q[2]);
          // bug? sum / 3?
          const double wgt = lookup_exp_[sum];
          double q_disp = plane_a * q_x + q_disp_y;
          int q_disp_floor = static_cast<int>(q_disp);
          if (q_disp_floor <= 0 || q_disp_floor >= max_disp_) {
            // impossible disparity --> very large cost
            cost += wgt * max_cost_[view];
          } else {
            // 1 - view --> other view
            int q_disp_ceil = q_disp_floor + 1;
            const double floor_wgt = q_disp_ceil - q_disp;
            double tmp =
              floor_wgt * *(cost_vol_[view][q_disp_floor].ptr<double>(q_y) +q_x)
              + (1 - floor_wgt) * *(cost_vol_[view][q_disp_ceil].ptr<double>(q_y) +q_x);
            cost += wgt * tmp;
          }
        } // end if q_x
      } // end for dx
    } // end if q_x
  } // end for dy
  return cost;
}
