#include "cs_patchmatch.h"


CSPatchMatch::CSPatchMatch(const Mat& l_img, const Mat& r_img, 
  const int& max_dis, const int& dis_scale) :
  max_dis_(max_dis), dis_scale_(dis_scale) {
  CV_Assert(l_img.type() == CV_64FC3 && r_img.type() == CV_64FC3);
  img_[kLeft]  = l_img.clone();
  img_[kRight] = r_img.clone();
  wid_ = img_[kLeft].cols;
  hei_ = img_[kLeft].rows;
  // allocate disparity maps
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
    dis_[v] = Mat::zeros(hei_, wid_, CV_8UC1);
  }
  // allocate plane and min_cost
  for (int v = 0; v < kViewNum; ++v) {
    plane_[v] = new Plane*[hei_];
    min_cost_[v] = new double*[hei_];
    for (int y = 0; y < hei_; ++y) {
      plane_[v][y] = new Plane[wid_];
      min_cost_[v][y] = new double[wid_];
      // set min cost to max double
      fill(min_cost_[v][y], min_cost_[v][y] + wid_, kDoubleMax);
    }
  }
}

CSPatchMatch::~CSPatchMatch() {
  for (int v = 0; v < kViewNum; ++v) {
    for (int y = 0; y < hei_; ++y) {
      delete[] plane_[v][y];
      delete[] min_cost_[v][y];
      plane_[v][y] = NULL;
      min_cost_[v][y] = NULL;
    }
    delete[] plane_[v];
    delete[] min_cost_[v];
    plane_[v] = NULL;
    min_cost_[v] = NULL;
  }
}

void CSPatchMatch::PatchMatch(const int& iter_num, 
  const IPlaneCost* plane_cost) {
  cout << "\t Patch Match" << endl;
  InitRandomPlane();
  for (int i = 0; i < iter_num; ++i) {
    cout << "\t\t Iter: " << i;
    SpatialPropagation(i, plane_cost);
    ViewPropagation(plane_cost);
    PlaneRefinement(max_dis_ / 2.0, kMaxNorm_, kZStopThres_, plane_cost);
  }
  PlaneToDisp();
  PostProcessing();
}

Mat& CSPatchMatch::dis(const RefView& view) {
  return dis_[view];
}

void CSPatchMatch::InitRandomPlane() {
  cout << "\t\t Init Random Plane" << endl;
  CV_Assert(plane_ != NULL);
  RNG rng;
  // paramter for gaussian distribution
  const Vec3d norm_avg(0.0, 0.0, 0.0);
  const Vec3d norm_std(1.0, 1.0, 1.0);
  for (int v = 0; v < kViewNum; ++v) {
    for (int y = 0; y < hei_; ++y) {
      for (int x = 0; x < wid_; ++x) {
        // rand point and norm
        double rand_dis = rng.uniform(kDoubleEps,
          static_cast<double>(max_dis_));
        plane_[v][y][x].set_point(Point3d(x, y, rand_dis));
        Vec3d rand_norm(0.0, 0.0, 0.0);
        rng.fill(rand_norm, RNG::NORMAL, norm_avg, norm_std);
        double denom = max(norm(rand_norm, NORM_L2), kDoubleEps);
        plane_[v][y][x].set_norm(rand_norm / denom);
        // udpate plane paramter
        plane_[v][y][x].update_param();
      }
    }
  }
}

///////////////////////////////////////////////////////
// Func: SpatialPropagation
// Desc: propagate plane params between neighbours.
//       even iteration --> top-left to bottom-right
//       odd iteration --> bottom-right to top-left
// In:
// const int& cur_iter -- iteration number
// const IPlaneCost* plane_cost -- iterface for computing
//                                 plane cost
//
// Out:
// void
///////////////////////////////////////////////////////
void CSPatchMatch::SpatialPropagation(const int& cur_iter,
  const IPlaneCost* plane_cost) {

  cout << "\t\t Spatial Propagation" << endl;
  // default to odd iteration
  int x_st = wid_ - 2, x_ed = -1, x_inc = -1;
  int y_st = hei_ - 2, y_ed = -1, y_inc = -1;
  if (cur_iter % 2 == 0) {
    // even iteration reverse order
    x_st = 1; x_ed = wid_; x_inc = 1;
    y_st = 1; y_ed = hei_; y_inc = 1;
  }
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1) ) {
    // row[y_st] has no y-axis neighbour
    for (int x = x_st; x != x_ed; x += x_inc) {
      const Plane& nx_plane = plane_[v][y_st - y_inc][x - x_inc];
      const double nx_cost =
        plane_cost->GetPlaneCost(x, y_st - y_inc, nx_plane, v);
      if (nx_cost < min_cost_[v][y_st - y_inc][x]) {
        min_cost_[v][y_st - y_inc][x] = nx_cost;
        plane_[v][y_st - y_inc][x] = nx_plane;
      }
    }
    for (int y = y_st; y != y_ed; y += y_inc) {
      for (int x = x_st; x != x_ed; x += x_inc) {
        // x-axis neighbour
        const Plane& nx_plane = plane_[v][y][x - x_inc];
        const double nx_cost =
          plane_cost->GetPlaneCost(x, y, nx_plane, v);
        if (nx_cost < min_cost_[v][y][x]) {
          min_cost_[v][y][x] = nx_cost;
          plane_[v][y][x] = nx_plane;
        }
        // y-axis neighbour
        const Plane& ny_plane = plane_[v][y - y_inc][x];
        const double ny_cost =
          plane_cost->GetPlaneCost(x, y, ny_plane, v);
        if (ny_cost < min_cost_[v][y][x]) {
          min_cost_[v][y][x] = ny_cost;
          plane_[v][y][x] = ny_plane;
        }
      }
    }
  } // end for view
}

///////////////////////////////////////////////////////
// Func: ViewPropagation
// Desc: propagate plane params across 2 views
// In:
// const IPlaneCost* plane_cost -- iterface for computing
//                                 plane cost
//
// Out:
// void
///////////////////////////////////////////////////////
void ViewPropagation(const IPlaneCost* plane_cost);
void CSPatchMatch::ViewPropagation(const IPlaneCost* plane_cost) {

  cout << "\t\t View Propagation" << endl;
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
    int other_view = 1 - v;
    // iterate all other view pixels
    for (int y = 0; y < hei_; ++y) {
      for (int x = 0; x < wid_; ++x) {
        Vec3d param = plane_[other_view][y][x].param();
        double disp = param[0] * x + param[1] * y + param[2];
        // get corresponding pixel in reference view
        int cor_x = 0;
        if (v == kLeft) {
          cor_x = HandleBorder(x + round(disp), wid_);
        } else {
          cor_x = HandleBorder(x - round(disp), wid_);
        }
        // update corresponding pixel's plane
        const Plane& cor_plane = plane_[other_view][y][x];
        const double cor_cost = 
          plane_cost->GetPlaneCost(cor_x, y, cor_plane, v);
        if (cor_cost < min_cost_[v][y][cor_x]) {
          min_cost_[v][y][cor_x] = cor_cost;
          plane_[v][y][cor_x] = cor_plane;
        }
      }
    }
  }
}

///////////////////////////////////////////////////////
// Func: PlaneRefinement
// Desc: random distrubing plane and update plane
// In:
// const double& z_max -- init maximum disturb on z
// const double& n_max -- init maximum disturb on norm
// const double& z_thres -- stop criteriion on z
// const IPlaneCost* plane_cost -- interface for
//                                 plane cost
//
// Out:
// void
///////////////////////////////////////////////////////
void CSPatchMatch::PlaneRefinement(const double& z_max, 
  const double& n_max, const double& z_thres,
  const IPlaneCost* plane_cost) {

  cout << "\t\t Plane Refinement" << endl;
  double z_iter = z_max;
  double n_iter = n_max;
  RNG rng;
  while (z_iter >= z_thres) {
    Vec3d n_low(-n_iter, -n_iter, -n_iter);
    Vec3d n_high(n_iter, n_iter, n_iter);
    for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
      for (int y = 0; y < hei_; ++y) {
        for (int x = 0; x < wid_; ++x) {
          Plane disturb_plane;
          Plane org_plane = plane_[v][y][x];
          Vec3d org_norm  = org_plane.norm();
          // distrub point (x, y, z)
          double disturb_z = org_plane.param().dot(Vec3d(x, y, 1.0));
          disturb_plane.set_point(
            Point3d(x, y,
              disturb_z + rng.uniform(-z_iter, z_iter)
             )
           );
          // distrub norm
          Vec3d delta_norm(0.0, 0.0, 0.0);
          rng.fill(delta_norm, RNG::UNIFORM, n_low, n_high);
          Vec3d disturb_norm = org_norm + delta_norm;
          double denom = max(norm(disturb_norm, NORM_L2),
            kDoubleEps);
          disturb_plane.set_norm(disturb_norm / denom);
          // re-calculate new param
          disturb_plane.update_param();
          // update plane
          const double distrub_cost = 
            plane_cost->GetPlaneCost(x, y, disturb_plane, v);
          if (distrub_cost < min_cost_[v][y][x]) {
            plane_[v][y][x] = disturb_plane;
            min_cost_[v][y][x] = distrub_cost;
          }
        }
      }
    }
    z_iter /= 2.0;
    n_iter /= 2.0;
  }
}

void CSPatchMatch::PostProcessing() {
  cout << "\t\t do nothing" << endl;
}

void CSPatchMatch::PlaneToDisp() {
  cout << "\t\t Convert Plane to Disparity" << endl;
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
    for (int y = 0; y < hei_; ++y) {
      uchar* dis_row = dis_[v].ptr<uchar>(y);
      for (int x = 0; x < wid_; ++x) {
        double disp = dis_scale_ *
          plane_[v][y][x].param().dot(Vec3d(x, y, 1.0));
        dis_row[x] = static_cast<uchar>(
          HandleBorder(static_cast<int>(disp), 256));
      }
    }
  }
}