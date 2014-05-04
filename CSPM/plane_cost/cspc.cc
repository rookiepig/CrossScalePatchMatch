///////////////////////////////////////////////////////
// File: cspc.cc
// Desc: corss scale cost aggregation
// 
// Author: rookiepig
// Date: 2014/04/03
//
///////////////////////////////////////////////////////
#include"cspc.h"

CSPC::CSPC(const Mat& l_img, const Mat& r_img,
  const int& max_disp, const int& wnd_size, const int& scale_num) :
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
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    lab_[v] = new Mat[scale_num_];
    grd_x_[v] = new Mat[scale_num_];
    for (int s = 0; s < scale_num_; ++s) {
      if (s > 0) {
        // pyramid down
        pyrDown(img_[v][s - 1], img_[v][s]);
        hei_[s] = (hei_[s - 1] + 1) / 2;
        wid_[s] = (wid_[s - 1] + 1) / 2;
        max_disp_[s] = max_disp_[s - 1] / 2;
      }
      // get LAB color image
      cvtColor(img_[v][s], lab_[v][s], CV_BGR2Lab);  
      // get x-axis gradient
      // Mat tmp;
      Mat gray;
      // img_[v].convertTo(tmp, CV_32F);
      cvtColor(img_[v][s], gray, CV_BGR2GRAY);
      // X Gradient
      // sobel size must be 1
      Sobel(gray, grd_x_[v][s], CV_16S, 1, 0, 1);
      // grd_x_[v] += 0.5;
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

CSPC::~CSPC(void) {
  // do nothing
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    delete[] img_[v];
    delete[] lab_[v];
    delete[] grd_x_[v];
  }
  delete[] scale_wgt_;
  delete[] lookup_exp_;
  delete[] wid_;
  delete[] hei_;
  delete[] max_disp_;
}

double CSPC::GetPlaneCost(const int& ref_x, const int& ref_y,
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
    for (int dy = -half_wnd; dy <= half_wnd; ++dy) {
      int q_y = HandleBorder(cur_y + dy, hei_[s]);
      for (int dx = -half_wnd; dx <= half_wnd; ++dx) {
        int q_x = HandleBorder(cur_x + dx, wid_[s]);
        const double wgt = 
          GetCostWeight(cur_x, cur_y, q_x, q_y, view, s);
        double q_disp = plane_param[0] * q_x +
          plane_param[1] * q_y + plane_param[2];
        if (q_disp <= 0.0 || q_disp >= max_disp_[s]) {
          // impossible disparity --> largest cost
          scale_cost += wgt * (COST_ALPHA * TAU_CLR +
            (1 - COST_ALPHA) * TAU_GRD);
        }
        else {
          //if (view == kLeft) {
          //  other_x = q_x - q_disp;
          //}
          //else {
          //  other_x = q_x + q_disp;
          //}
          const double other_x = q_x + (2 * view - 1) * q_disp;
          scale_cost += wgt * 
            GetPixelCost(q_x, q_y, other_x, q_y, view, s);
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

inline double CSPC::GetCostWeight(const int& ref_x,
  const int& ref_y, const int& q_x, const int& q_y,
  const RefView& view, const int& scale) const {
  // assume three channel
  const uchar* I_p = lab_[view][scale].ptr<uchar>(ref_y) + 3 * ref_x;
  const uchar* I_q = lab_[view][scale].ptr<uchar>(q_y) + 3 * q_x;
  int sum = abs(I_p[0] - I_q[0]) +
            abs(I_p[1] - I_q[1]) +
            abs(I_p[2] - I_q[2]);
  return lookup_exp_[sum];
  // return exp(-sum / gamma_);
}

inline double CSPC::GetPixelCost(const int& ref_x, 
  const int& ref_y, const double& other_x, const int& other_y,
  const RefView& view, const int& scale) const {
 
  int floor_x = static_cast<int>(Floor2Int(other_x));
  int ceil_x = floor_x + 1;
  const double floor_wgt = ceil_x - other_x;
  // const double ceil_wgt = 1 - floor_wgt;
  // handle special border
  floor_x = HandleBorder(floor_x, wid_[scale]);
  ceil_x  = HandleBorder(ceil_x, wid_[scale]);

  // cout << " floor_x: " << floor_x << " ceil_x: " << ceil_x <<
  //  " ref_y: " << ref_y << " other_y: " << other_y << endl;
  const uchar* I_q = img_[view][scale].ptr<uchar>(ref_y) + 3 * ref_x;
  // 1 - view --> other view
  const uchar* I_ohter_y = img_[1 - view][scale].ptr<uchar>(other_y);
  const uchar* I_floor = I_ohter_y + 3 * floor_x;
  const uchar* I_ceil = I_ohter_y + 3 * ceil_x;
  // interpolated color difference
  double clr_cost = 
    fabs(I_q[0] - I_ceil[0] + floor_wgt * (I_ceil[0] - I_floor[0])) +
    fabs(I_q[1] - I_ceil[1] + floor_wgt * (I_ceil[1] - I_floor[1])) +
    fabs(I_q[2] - I_ceil[2] + floor_wgt * (I_ceil[2] - I_floor[2]));
  // for (int c = 0; c < 3; ++c) {
  // } 
  const short G_q = *(grd_x_[view][scale].ptr<short>(ref_y) +ref_x);
  const short* G_other_y = grd_x_[1 - view][scale].ptr<short>(other_y);
  const short G_floor = G_other_y[floor_x];
  const short G_ceil = G_other_y[ceil_x];
  // interpolated gradient difference
  double grd_cost = 
    fabs(G_q - G_ceil + floor_wgt * (G_ceil - G_floor));

  clr_cost = clr_cost > TAU_CLR ? TAU_CLR : clr_cost;
  grd_cost = grd_cost > TAU_GRD ? TAU_GRD : grd_cost;
  return COST_ALPHA * clr_cost + (1 - COST_ALPHA) * grd_cost;
}