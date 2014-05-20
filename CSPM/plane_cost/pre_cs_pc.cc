///////////////////////////////////////////////////////
// File: pre_cs_pc.cc
// Desc: precompute cost volume
//       for cross-scale plane cost
//
// Author: rookiepig
// Date: 2014/05/19
//
///////////////////////////////////////////////////////
#include"pre_cs_pc.h"

PreCSPC::PreCSPC(const Mat& l_img, const Mat& r_img,
  const int& max_disp, const int& wnd_size,
  const int& scale_num, CCMethod* cc_method) :
  //const double& alpha,
  //const double& tau_clr, const double& tau_grd,
  //const double& gamma) :
  wnd_size_(wnd_size), scale_num_(scale_num) {
  //alpha_(alpha),
  //tau_clr_(tau_clr), tau_grd_(tau_grd),
  //gamma_(gamma) {

  // for TAD + Grd input image must be CV_64FC3
  // CV_Assert(l_img.type() == CV_64FC3 && r_img.type() == CV_64FC3);
  CV_Assert(l_img.type() == CV_8UC3 && r_img.type() == CV_8UC3);
  cout << "\t cross scale plane cost\n";
  img_[kLeft] = new Mat[scale_num_];
  img_[kLeft][0]  = l_img.clone();
  img_[kRight] = new Mat[scale_num_];
  img_[kRight][0] = r_img.clone();
  hei_ = new int[scale_num_];
  wid_ = new int[scale_num_];
  hei_[0] = l_img.rows;
  wid_[0] = l_img.cols;
  max_disp_ = new int[scale_num_];
  max_disp_[0] = max_disp;
  half_wnd_ = wnd_size_ / 2;
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    // init cost volum data
    cost_vol_[v] = new Mat*[scale_num_];
    max_cost_[v] = new double[scale_num_];
    for (int s = 0; s < scale_num_; ++s) {
      if (s > 0) {
        // pyramid down
        pyrDown(img_[v][s - 1], img_[v][s]);
        hei_[s] = (hei_[s - 1] + 1) / 2;
        wid_[s] = (wid_[s - 1] + 1) / 2;
        max_disp_[s] = max_disp_[s - 1] / 2;
      }
      cost_vol_[v][s] = new Mat[max_disp_[s] + 1];
      for (int d = 0; d < max_disp_[s] + 1; ++d) {
        cost_vol_[v][s][d] = Mat::zeros(hei_[s], wid_[s], CV_64FC1);
      }
    }
  }
  // construct multi-scale cost volume
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    for (int s = 0; s < scale_num_; ++s) {
      // convert left and right image for cost volume
      Mat tmp_l, tmp_r;
      cvtColor(img_[kLeft][s], tmp_l, CV_BGR2RGB);
      cvtColor(img_[kRight][s], tmp_r, CV_BGR2RGB);
      tmp_l.convertTo(tmp_l, CV_64F);
      tmp_r.convertTo(tmp_r, CV_64F);

      // build cost volume
      if (v == kLeft) {
        cc_method->buildCV(tmp_l, tmp_r, 
          max_disp_[s] + 1, cost_vol_[v][s]);
      } else if (v == kRight) {
        cc_method->buildRightCV(tmp_l, tmp_r, 
          max_disp_[s] + 1, cost_vol_[v][s]);
      }
      // record maximum cost value
      max_cost_[v][s] = -1.0;
      for (int d = 0; d < max_disp_[s] + 1; ++d) {
        double tmp = 0.0;
        minMaxLoc(cost_vol_[v][s][d], NULL, &tmp);
        if (tmp > max_cost_[v][s]) {
          max_cost_[v][s] = tmp;
        }
      }
    }
  }
  // multi-scale weight
  cout << "\t\t Reg param: " << REG_LAMBDA << endl;
	// construct regularization matrix
	Mat regMat = Mat::zeros(scale_num_, scale_num_, CV_64FC1);
	for (int s = 0; s < scale_num_; ++s) {
		if (s == 0) {
			regMat.at<double>(s, s) = 1 + REG_LAMBDA;
			regMat.at<double>(s, s + 1) = -REG_LAMBDA;
		}
		else if (s == scale_num_ - 1) {
			regMat.at<double>(s, s) = 1 + REG_LAMBDA;
			regMat.at<double>(s, s - 1) = -REG_LAMBDA;
		}
		else {
			regMat.at<double>(s, s) = 1 + 2 * REG_LAMBDA;
			regMat.at<double>(s, s - 1) = -REG_LAMBDA;
			regMat.at<double>(s, s + 1) = -REG_LAMBDA;
		}
	}
	Mat regInv = regMat.inv();
	scale_wgt_ = new double[scale_num_];
	for (int s = 0; s < scale_num_; ++s) {
		scale_wgt_[s] = regInv.at<double>(0, s);
	}
  // init exp look-up table
  lookup_exp_ = new double[1000];
  for (int i = 0; i < 1000; ++i) {
    lookup_exp_[i] = exp(-i * 1.0 / WGT_GAMMA);
  }
}

PreCSPC::~PreCSPC(void) {
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    delete[] img_[v];
    for(int s = 0; s < scale_num_; ++s) {
      delete[] cost_vol_[v][s];
    }
    delete[] cost_vol_[v];
    delete[] max_cost_[v];
  }
  delete[] scale_wgt_;
  delete[] lookup_exp_;
  delete[] wid_;
  delete[] hei_;
  delete[] max_disp_;
}

double PreCSPC::GetPlaneCost(const int& ref_x, const int& ref_y,
  const Plane& plane, const RefView& view) const {
  const int half_wnd = wnd_size_ / 2;
  double cost = 0.0;
  Vec3d org_norm = plane.norm();
  Vec3d plane_param = plane.param();
  double cur_disp = plane_param[0] * ref_x + plane_param[1] * ref_y +
    plane_param[2];
  int cur_y = ref_y;
  int cur_x = ref_x;
  for (int s = 0; s < scale_num_; ++s) {
    Plane cur_plane(org_norm, Point3d(cur_x, cur_y, cur_disp));
    Vec3d plane_param = cur_plane.param();
    double scale_cost = 0.0f;
    const double plane_a = plane_param[0];
    const double plane_b = plane_param[1];
    const double plane_c = plane_param[2];
    const uchar* I_p = img_[view][s].ptr<uchar>(cur_y) +3 * cur_x;
    for (int dy = -half_wnd; dy <= half_wnd; ++dy) {
      int q_y = cur_y + dy;
      if (q_y >= 0 && q_y < hei_[s]) {
        const uchar* I_q_y = img_[view][s].ptr<uchar>(q_y);
        const double q_disp_y = plane_b * q_y + plane_c;

        for (int dx = -half_wnd; dx <= half_wnd; ++dx) {
          int q_x = cur_x + dx;
          if (q_x >= 0 && q_x < wid_[s]) {
            const uchar* I_q = I_q_y + 3 * q_x;
            int sum = abs(I_p[0] - I_q[0]) +
              abs(I_p[1] - I_q[1]) +
              abs(I_p[2] - I_q[2]);
            const double wgt = lookup_exp_[sum];
            double q_disp = plane_a * q_x + q_disp_y;
            int q_disp_floor = static_cast<int>(q_disp);
            if (q_disp_floor <= 0 || q_disp_floor >= max_disp_[s]) {
              // impossible disparity --> largest cost
              scale_cost += wgt * max_cost_[view][s];
            } else {
              int q_disp_ceil = q_disp_floor + 1;
              const double floor_wgt = q_disp_ceil - q_disp;
              double tmp =
              floor_wgt * *(cost_vol_[view][s][q_disp_floor].ptr<double>(q_y) +q_x)
              + (1 - floor_wgt) * *(cost_vol_[view][s][q_disp_ceil].ptr<double>(q_y) +q_x);
              scale_cost += wgt * tmp;
            }
          }
        }
      }
    }
    cost += scale_cost * scale_wgt_[s];
    cur_y /= 2;
    cur_x /= 2;
    cur_disp /= 2.0;
  }
  return cost;
}