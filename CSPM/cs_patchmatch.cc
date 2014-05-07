#include "cs_patchmatch.h"


CSPatchMatch::CSPatchMatch(const Mat& l_img, const Mat& r_img, 
  const int& max_dis, const int& dis_scale) :
  max_dis_(max_dis), dis_scale_(dis_scale) {

  // CV_Assert(l_img.type() == CV_64FC3 && r_img.type() == CV_64FC3);
  CV_Assert(l_img.type() == CV_8UC3 && r_img.type() == CV_8UC3);
  img_[kLeft]  = l_img.clone();
  img_[kRight] = r_img.clone();
  wid_ = img_[kLeft].cols;
  hei_ = img_[kLeft].rows;
  // allocate disparity maps
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
    dis_[v] = Mat::zeros(hei_, wid_, CV_8UC1);
  }
  // allocate plane and min_cost
#ifdef USE_POINTER
  plane_    = new Plane[kViewNum * hei_ * wid_];
  min_cost_ = new double[kViewNum * hei_ * wid_];
#else
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
#endif
}

CSPatchMatch::~CSPatchMatch() {
#ifdef USE_POINTER
  delete[] plane_;
  delete[] min_cost_;
#else
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
#endif
}
void CSPatchMatch::PatchMatch(const int& iter_num, 
  const IPlaneCost* plane_cost) {
  cout << "\t Patch Match" << endl;

  InitRandomPlane(plane_cost);
#ifdef MY_DEBUG
  const int view_x = 300;
  const int view_y = 30;
  CV_Assert(view_x >= 0 && view_x < wid_ && view_y >= 0 && view_y < hei_);
  // view intermediate results
  PrintPixelInfo(view_x, view_y);
  ViewDisp();
#endif

  for (int i = 0; i < iter_num; ++i) {
    cout << "\t\t Iter: " << i << endl;
#ifdef MY_DEBUG
    // init time
    double duration = static_cast<double>(getTickCount());
#endif

    SpatialPropagation(i, plane_cost);

#ifdef MY_DEBUG
    // record time
    duration = static_cast<double>(getTickCount()) - duration;
    duration /= cv::getTickFrequency(); // the elapsed time in sec
    cout << "\t\t spatial prop time: " << duration << endl;
#endif

#ifdef MY_DEBUG
    // view intermediate results
    PrintPixelInfo(view_x, view_y);
    ViewDisp();
#endif

    ViewPropagation(i, plane_cost);

#ifdef MY_DEBUG
    // view intermediate results
    PrintPixelInfo(view_x, view_y);
    ViewDisp();
#endif
    PlaneRefinement(max_dis_ / 2.0, kMaxNorm_, kZStopThres_, plane_cost);
#ifdef MY_DEBUG
    // view intermediate results
    PrintPixelInfo(view_x, view_y);
    ViewDisp();
#endif
  }
  PlaneToDisp();
  //for (int i = 0; i < 3; ++i) {
  //PostProcessing();
  //}
}

Mat& CSPatchMatch::dis(const RefView& view) {
  return dis_[view];
}

void CSPatchMatch::InitRandomPlane(const IPlaneCost* plane_cost) {
  cout << "\t\t Init Random Plane" << endl;
  CV_Assert(plane_ != NULL);
  // random generator
#ifdef MY_DEUBG
  RNG rng;   // use same seed for easy debug
#else
  RNG rng(time(NULL));
#endif
  // paramter for gaussian distribution
  const double norm_avg = 0.0;
  const double norm_std = 1.0;
#ifdef USE_POINTER
  Plane* cur_plane = plane_;
  double* cur_min_cost = min_cost_;
#endif
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
#ifndef MY_DEBUG
    // turn off omp when debugging
    #pragma omp parallel for
#endif
    for (int y = 0; y < hei_; ++y) {
      for (int x = 0; x < wid_; ++x) {
        // rand point and norm
        double rand_dis = rng.uniform(kDoubleEps,
          static_cast<double>(max_dis_));
#ifdef USE_POINTER
        cur_plane->set_point(Point3d(x, y, rand_dis));
#else
        plane_[v][y][x].set_point(Point3d(x, y, rand_dis));
#endif
        Vec3d rand_norm(0.0, 0.0, 0.0);
        rng.fill(rand_norm, RNG::NORMAL, norm_avg, norm_std);
        double denom = max(norm(rand_norm, NORM_L2), kDoubleEps);
#ifdef USE_POINTER
        cur_plane->set_norm(rand_norm / denom);
        // udpate plane paramter
        cur_plane->update_param();
        *cur_min_cost = 
          plane_cost->GetPlaneCost(x, y, *cur_plane, v);
        ++cur_plane;
        ++cur_min_cost;
#else
        plane_[v][y][x].set_norm(rand_norm / denom);
        // udpate plane paramter
        plane_[v][y][x].update_param();
        min_cost_[v][y][x] = 
          plane_cost->GetPlaneCost(x, y, plane_[v][y][x], v);
#endif
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
    cout << "\t\t\t view: " << v << endl;
#ifdef USE_POINTER
    Plane* cur_plane = plane_ + (v + 1) * hei_ * wid_ - 1;
    double* cur_min_cost = min_cost_ + (v + 1) * hei_ * wid_ - 1;
    if (cur_iter % 2 == 0) {
      cur_plane = plane_ + v * hei_ * wid_;
      cur_min_cost = min_cost_ + v * hei_ * wid_;
    }
    cur_plane += x_inc;
    cur_min_cost += x_inc;
#endif
    // specially handle the first row, i.e.  y_st - y_inc
    for (int x = x_st; x != x_ed; x += x_inc) {
#ifdef USE_POINTER
      const Plane& nx_plane = *(cur_plane - x_inc);
      const double nx_cost =
        plane_cost->GetPlaneCost(x, y_st - y_inc, nx_plane, v);
      if (nx_cost < *cur_min_cost) {
        *cur_min_cost = nx_cost;
        *cur_plane = nx_plane;
      }
      cur_plane += x_inc;
      cur_min_cost += x_inc;
#else
      const Plane& nx_plane = plane_[v][y_st - y_inc][x - x_inc];
      const double nx_cost =
        plane_cost->GetPlaneCost(x, y_st - y_inc, nx_plane, v);
      if (nx_cost < min_cost_[v][y_st - y_inc][x]) {
        min_cost_[v][y_st - y_inc][x] = nx_cost;
        plane_[v][y_st - y_inc][x] = nx_plane;
      }
#endif
    }
    for (int y = y_st; y != y_ed; y += y_inc) {
#ifdef USE_POINTER
      cur_plane += x_inc;
      cur_min_cost += x_inc;
#endif
      for (int x = x_st; x != x_ed; x += x_inc) {
        // x-axis neighbour
#ifdef USE_POINTER
        const Plane& nx_plane = *(cur_plane - x_inc);
#else
        const Plane& nx_plane = plane_[v][y][x - x_inc];
#endif
        const double nx_cost =
          plane_cost->GetPlaneCost(x, y, nx_plane, v);
#ifdef USE_POINTER
        if (nx_cost < *cur_min_cost) {
          *cur_min_cost = nx_cost;
          *cur_plane = nx_plane;
        }
#else
        if (nx_cost < min_cost_[v][y][x]) {
          min_cost_[v][y][x] = nx_cost;
          plane_[v][y][x] = nx_plane;
        }
#endif
        // y-axis neighbour
#ifdef USE_POINTER
        const Plane& ny_plane = *(cur_plane - y_inc * wid_);
#else
        const Plane& ny_plane = plane_[v][y - y_inc][x];
#endif
        const double ny_cost =
          plane_cost->GetPlaneCost(x, y, ny_plane, v);
#ifdef USE_POINTER
        if (ny_cost < *cur_min_cost) {
          *cur_min_cost = ny_cost;
          *cur_plane = ny_plane;
        }
        cur_plane += x_inc;
        cur_min_cost += x_inc;
#else
        if (ny_cost < min_cost_[v][y][x]) {
          min_cost_[v][y][x] = ny_cost;
          plane_[v][y][x] = ny_plane;
        }
#endif
      }
    }
  } // end for view
}

///////////////////////////////////////////////////////
// Func: ViewPropagation
// Desc: propagate plane params across 2 views
// In:
// const int& cur_iter -- iteration number
// const IPlaneCost* plane_cost -- iterface for computing
//                                 plane cost
//
// Out:
// void
///////////////////////////////////////////////////////
void CSPatchMatch::ViewPropagation(const int& cur_iter,
  const IPlaneCost* plane_cost) {

  cout << "\t\t View Propagation" << endl;
  Plane cor_plane;
  // default to odd iteration
  int x_st = wid_ - 1, x_ed = -1, x_inc = -1;
  int y_st = hei_ - 1, y_ed = -1, y_inc = -1;
  if (cur_iter % 2 == 0) {
    // even iteration reverse order
    x_st = 0; x_ed = wid_; x_inc = 1;
    y_st = 0; y_ed = hei_; y_inc = 1;
  }
#ifdef USE_POINTER
  Plane* current_plane = plane_;
  double* current_min_cost = min_cost_;
#endif
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
    int other_view = 1 - v;
#ifdef USE_POINTER
    Plane* other_plane = plane_ + other_view * hei_ * wid_;
#endif
    cout << "\t\t\t view: " << v << endl;
    // iterate all other view pixels
    for (int y = y_st; y != y_ed; y += y_inc) {
      for (int x = x_st; x != x_ed; x += x_inc) {
#ifdef USE_POINTER
        Vec3d param = other_plane->param();
#else
        Vec3d param = plane_[other_view][y][x].param();
#endif
        double disp = param[0] * x + param[1] * y + param[2];
        // get corresponding pixel in reference view
        int cor_x = 0;
#ifdef USE_POINTER
        if (v == kLeft) {
          cor_x = HandleBorder(x + Round2Int(disp), wid_);
        } else {
          cor_x = HandleBorder(x - Round2Int(disp), wid_);
        }
        // set corresponding pixel's plane
        cor_plane.set_norm(other_plane->norm());
        cor_plane.set_point(
          Point3d(cor_x, y, disp));
        cor_plane.update_param();
        const double cor_cost = 
          plane_cost->GetPlaneCost(cor_x, y, cor_plane, v);
        if (cor_cost < current_min_cost[cor_x]) {
          current_min_cost[cor_x] = cor_cost;
          current_plane[cor_x] = cor_plane;
        }
        ++other_plane;
#else
        if (v == kLeft) {
          // cor_x in left view
          cor_x = HandleBorder(x + Round2Int(disp), wid_);
        }
        else {
          cor_x = HandleBorder(x - Round2Int(disp), wid_);
        }
        // set corresponding pixel's plane
        cor_plane.set_norm(plane_[other_view][y][x].norm());
        cor_plane.set_point(
          Point3d(cor_x, y, disp));
        cor_plane.update_param();
        const double cor_cost =
          plane_cost->GetPlaneCost(cor_x, y, cor_plane, v);
        if (cor_cost < min_cost_[v][y][cor_x]) {
          min_cost_[v][y][cor_x] = cor_cost;
          plane_[v][y][cor_x] = cor_plane;
        }
#endif
      } // for x
#ifdef USE_POINTER
      current_plane += wid_;
      current_min_cost += wid_;
#endif
    } // for y
  } // for view
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
  while (z_iter >= z_thres) {
    cout << "\t\t\t pf iter cur z_max: " << z_iter << endl;
#ifdef USE_POINTER
    Plane* cur_plane = plane_;
    double* cur_min_cost = min_cost_;
#endif
    // random generator
#ifdef MY_DEBUG
    RNG rng;    // use same seed
#else
    RNG rng(time(NULL));
#endif
    for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
#ifndef MY_DEBUG
      #pragma omp parallel for private(rng)
#endif
      for (int y = 0; y < hei_; ++y) {
        for (int x = 0; x < wid_; ++x) {
          // random disturbing variables
          Plane disturb_plane;
          Vec3d delta_norm(0.0, 0.0, 0.0);
          Vec3d disturb_norm(0.0, 0.0, 0.0);
#ifdef USE_POINTER
          Vec3d org_plane_param = cur_plane->param();
#else
          Vec3d org_plane_param = plane_[v][y][x].param();
#endif
          // distrub point (x, y, z)
          double disturb_z = org_plane_param[0] * x +
                             org_plane_param[1] * y +
                             org_plane_param[2];
          disturb_plane.set_point(
            Point3d(x, y, disturb_z + rng.uniform(-z_iter, z_iter))
           );
          // distrub norm
          rng.fill(delta_norm, RNG::UNIFORM, -n_iter, n_iter);
#ifdef USE_POINTER
          disturb_norm = cur_plane->norm() + delta_norm;
#else
          disturb_norm = plane_[v][y][x].norm() + delta_norm;
#endif
          double denom = max(norm(disturb_norm, NORM_L2),
            kDoubleEps);
          disturb_plane.set_norm(disturb_norm / denom);
          // re-calculate new param
          disturb_plane.update_param();
          // update plane
          const double distrub_cost = 
            plane_cost->GetPlaneCost(x, y, disturb_plane, v);
#ifdef USE_POINTER
          if (distrub_cost < *cur_min_cost) {
            *cur_plane = disturb_plane;
            *cur_min_cost = distrub_cost;
          }
          ++cur_plane;
          ++cur_min_cost;
#else
          if (distrub_cost < min_cost_[v][y][x]) {
            plane_[v][y][x] = disturb_plane;
            min_cost_[v][y][x] = distrub_cost;
          }
#endif
        }
      }
    }
    z_iter /= 2.0;
    n_iter /= 2.0;
  }
}

void CSPatchMatch::LeftRightCheck(int** valid) {
  cout << "\t\t\t left-right check" << endl;
  int* l_valid = valid[kLeft];
  int* r_valid = valid[kRight];
  for (int y = 0; y < hei_; y++) {
    uchar* l_dis_data = dis_[kLeft].ptr<uchar>(y);
    uchar* r_dis_data = dis_[kRight].ptr<uchar>(y);
    for (int x = 0; x < wid_; x++) {
      // check left image
      double l_disp = l_dis_data[x] * 1.0 / dis_scale_;
      // assert( ( x - lDep ) >= 0 && ( x - lDep ) < wid );
      int r_loc = (x - static_cast<int>(l_disp) + wid_) % wid_;
      CV_Assert(r_loc >= 0 && r_loc < wid_);
      double r_disp = r_dis_data[r_loc] * 1.0 / dis_scale_;
      // disparity should not be zero
      if (fabs(l_disp - r_disp) <= 1.0 && l_disp > 0) {
        *l_valid = 1;
      }
      // check right image
      r_disp = r_dis_data[x] / dis_scale_;
      // assert( ( x + rDep ) >= 0 && ( x + rDep ) < wid );
      int l_loc = (x + static_cast<int>(r_disp) + wid_) % wid_;
      CV_Assert(l_loc >= 0 && l_loc < wid_);
      l_disp = l_dis_data[l_loc] * 1.0 / dis_scale_;
      // disparity should not be zero
      if (fabs(r_disp - l_disp) <= 1.0 && r_disp > 0) {
        *r_valid = 1;
      }
      ++l_valid;
      ++r_valid;
    }
  }
}
void CSPatchMatch::FillInvalid(int** valid) {
  cout << "\t\t\t fill invalid pixel" << endl;
#ifdef USE_POINTER
  Plane* cur_plane = plane_;
#endif
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    int* cur_valid = valid[v];
    for (int y = 0; y < hei_; y++) {
      int* y_valid = valid[v] + y * wid_;
      uchar* dis_data = dis_[v].ptr<uchar>(y);
      for (int x = 0; x < wid_; x++) {
        if (*cur_valid == 0) {
          // find left first valid pixel
          int l_first = x;
          int l_find = 0;
          while (l_first >= 0) {
            if (y_valid[l_first]) {
              l_find = 1;
              break;
            }
            --l_first;
          }
          int r_find = 0;
          // find right first valid pixel
          int r_first = x;
          while (r_first < wid_) {
            if (y_valid[r_first]) {
              r_find = 1;
              break;
            }
            ++r_first;
          }
          // set x's depth to the lowest one
          if (l_find && r_find) {
#ifdef USE_POINTER
            double l_first_disp =
              cur_plane[l_first].param().dot(Vec3d(x, y, 1.0));
            double r_first_disp = 
              cur_plane[r_first].param().dot(Vec3d(x, y, 1.0));
#else
            double l_first_disp =
              plane_[v][y][l_first].param().dot(Vec3d(x, y, 1.0));
            double r_first_disp =
              plane_[v][y][r_first].param().dot(Vec3d(x, y, 1.0));
#endif
            if (l_first_disp <= r_first_disp) {
              l_first_disp *= dis_scale_;
              dis_data[x] = static_cast<uchar>(
                HandleBorder(static_cast<int>(l_first_disp), 256)
               );
            } else {
              r_first_disp *= dis_scale_;
              dis_data[x] = static_cast<uchar>(
                HandleBorder(static_cast<int>(r_first_disp), 256)
               );
            }
          } else if (l_find) {
#ifdef USE_POINTER
            double l_first_disp =
              cur_plane[l_first].param().dot(Vec3d(x, y, 1.0));
#else
            double l_first_disp =
              plane_[v][y][l_first].param().dot(Vec3d(x, y, 1.0));
#endif
            l_first_disp *= dis_scale_;
            dis_data[x] = static_cast<uchar>(
              HandleBorder(static_cast<int>(l_first_disp), 256)
             );
          } else if (r_find) {
#ifdef USE_POINTER
            double r_first_disp = 
              cur_plane[r_first].param().dot(Vec3d(x, y, 1.0));
#else
            double r_first_disp =
              plane_[v][y][r_first].param().dot(Vec3d(x, y, 1.0));
#endif
            r_first_disp *= dis_scale_;
            dis_data[x] = static_cast<uchar>(
              HandleBorder(static_cast<int>(r_first_disp), 256)
             );
          } // end if l_find && r_find
        } // end if cur_valid
        cur_valid++;
      } // end for x
#ifdef USE_POINTER
      cur_plane += wid_;
#endif
    } // end for y
  } // end for each view
}

void CSPatchMatch::WeightedMedian(int** valid, 
  const int wnd_size, const double gamma) {
  int half_wnd = wnd_size / 2;
  // init exp look-up table
  double* lookup_exp = new double[1000];
  for (int i = 0; i < 1000; ++i) {
    lookup_exp[i] = exp(-i / gamma);
  }
  double* disp_hist = new double[256];

  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
    // filter left
    int* cur_valid = valid[v];
    for (int y = 0; y < hei_; y++) {
      uchar* l_dis_data = dis_[v].ptr<uchar>(y);
      const uchar* pL = img_[v].ptr<uchar>(y);
      for (int x = 0; x < wid_; x++) {
        if (*cur_valid == 0) {
          const uchar* pL_x = pL + 3 * x;
          // just filter invalid pixels
          fill(disp_hist, disp_hist + 256, 0.0);
          double sum_wgt = 0.0f;
          // set disparity histogram by bilateral weight
          for (int wy = -half_wnd; wy <= half_wnd; wy++) {
            const int qy = HandleBorder(y + wy, hei_);
            // int* qLValid = lValid + qy * wid;
            const uchar* qL = img_[v].ptr<uchar>(qy);
            const uchar* q_dis_data = dis_[v].ptr<uchar>(qy);
            for (int wx = -half_wnd; wx <= half_wnd; wx++) {
              const int qx = HandleBorder(x + wx, wid_);
              // invalid pixel also used
              // if( qLValid[ qx ] && wx != 0 && wy != 0 ) {
              const int q_disp = static_cast<int>(q_dis_data[qx]);
              if (q_disp > 0) {
                // double disWgt = wx * wx + wy * wy;
                // disWgt = sqrt( disWgt );
                const uchar * qL_x = qL + 3 * qx;
                int clr_diff =
                  abs(pL_x[0] - qL_x[0]) +
                  abs(pL_x[1] - qL_x[1]) +
                  abs(pL_x[2] - qL_x[2]);
                // clrWgt = sqrt( clrWgt );
                double wgt = lookup_exp[clr_diff];
                disp_hist[q_disp] += wgt;
                sum_wgt += wgt;
              }
              // }
            }
          }
          double median_wgt = sum_wgt / 2.0;
          sum_wgt = 0.0;
          int median_disp = 0;
          for (int d = 0; d < 256; d++) {
            sum_wgt += disp_hist[d];
            if (sum_wgt >= median_wgt) {
              median_disp = d;
              break;
            }
          }
          // set new disparity
          l_dis_data[x] = median_disp;
        }
        cur_valid++;
      }
    }
  }
  delete[] disp_hist;
  delete[] lookup_exp;
}

void CSPatchMatch::PostProcessing() {
  cout << "\t\t Weighted Median Post Processing" << endl;
  int** valid = new int*[kViewNum];
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    valid[v] = new int[hei_ * wid_]();    // set to 0 (invalid)
  }
  // left-right check
  LeftRightCheck(valid);

// #define VIEW_PP
#ifdef VIEW_PP
  // visualize valid
  Mat tmp(hei_, wid_, CV_8UC1);
  int* p_valid = valid[kLeft];
  for (int y = 0; y < hei_; ++y) {
    for (int x = 0; x < wid_; ++x) {
      if (*p_valid) {
        tmp.at<uchar>(y, x) = 255;
      } else {
        tmp.at<uchar>(y, x) = 0;
      }
      ++p_valid;
    }
  }
   imshow("l_valid", tmp);
   imwrite("l_valid.png", tmp);
   imshow("l_raw", dis_[kLeft]);
   imwrite("l_raw.png", dis_[kLeft]);
   waitKey(-1);
#endif

  // fill invalid
  FillInvalid(valid);

#ifdef VIEW_PP
   imshow("l_fill", dis_[kLeft]);
   imwrite("l_fill.png", dis_[kLeft]);
   waitKey(-1);
#endif

  // weighted median filter
  const int wnd_size = 35;
  const double gamma = 10.0;
  WeightedMedian(valid, wnd_size, gamma);

#ifdef VIEW_PP
  imshow("l_final", dis_[kLeft]);
  waitKey(-1);
#endif

  // release valid flag
  for (int v = kLeft; v <= kRight; v = RefView(v + 1)) {
    delete[] valid[v];
    valid[v] = NULL;
  }
  delete[] valid;
  valid = NULL;
}

void CSPatchMatch::PlaneToDisp() {
  cout << "\t\t Convert Plane to Disparity" << endl;
#ifdef USE_POINTER
  Plane* cur_plane = plane_;
#endif
  for (RefView v = kLeft; v <= kRight; v = RefView(v + 1)) {
    for (int y = 0; y < hei_; ++y) {
      uchar* dis_row = dis_[v].ptr<uchar>(y);
      for (int x = 0; x < wid_; ++x) {
#ifdef USE_POINTER
        double disp = cur_plane->param().dot(Vec3d(x, y, 1.0));
        ++cur_plane;
#else
        double disp = plane_[v][y][x].param().dot(Vec3d(x, y, 1.0));
#endif
        // narrow disp to [0, 255]
        dis_row[x] = saturate_cast<uchar>(Round2Int(disp * dis_scale_));
      }
    }
  }
}