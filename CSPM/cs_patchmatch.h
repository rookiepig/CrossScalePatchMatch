///////////////////////////////////////////////////////
// File: cs_patchmatch.h
// Desc: cross scale patch match
// 
// Author: rookiepig
// Date: 204/04/01
//
///////////////////////////////////////////////////////
#pragma once
#include"commfunc.h"
#include"plane.h"
#include"plane_cost\i_plane_cost.h"

#define WMF_GAMMA 10.0

class CSPatchMatch
{
 public:

   CSPatchMatch(const Mat& l_img, const Mat& r_img,
     const int& max_dis, const int& dis_scale);

  ~CSPatchMatch();

#ifdef MY_DEBUG
  // function for debug usage
  void ViewDisp(void) {
    // view disparity
    PlaneToDisp();
    imshow("l_dis", dis_[kLeft]);
    imshow("r_dis", dis_[kRight]);
    cout << "press any key..." << endl;
    waitKey(-1);
  }
  void PrintPixelInfo(const int& x, const int& y) {
    // print pixel information
    Plane plane = plane_[kLeft][y][x];
    double min_cost = min_cost_[kLeft][y][x];
    cout << "Pixel ( " << x << ", " << y << " ):" << endl;
    cout << "\t min_cost: " << min_cost << endl;
    cout << "\t disp: " << plane.param().dot(Vec3d(x, y, 1.0)) << endl;
    Vec3d norm = plane.norm();
    cout << "\t norm: " << " (" 
         << norm[0] << " " 
         << norm[1] << " "
         << norm[2] << " )" << endl;
    Point3d point = plane.point();
    cout << "\t point: " << " (" 
         << point.x << " " 
         << point.y << " "
         << point.z << " )" << endl;
  }
#endif
  ///////////////////////////////////////////////////////
  // Func: PatchMatch
  // Desc: perform patch match
  // In:
  // const int& iter_num -- iteration number
  // const IPlaneCost* plane_cost -- interface for
  //                                 computing plane cost
  //
  // Out:
  // OutputParam
  ///////////////////////////////////////////////////////
  void PatchMatch(const int& iter_num, 
    const IPlaneCost* plane_cost, const bool& use_pp);

  Mat& dis(const RefView& view);

 private:

   void InitRandomPlane(const IPlaneCost* plane_cost);

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
   void SpatialPropagation(const int& cur_iter, 
     const IPlaneCost* plane_cost);

   ///////////////////////////////////////////////////////
   // Func: ViewPropagation
   // Desc: propagate plane params across 2 views
   // In:
   // const int& cur_iter -- iteration number
   // const IPlaneCost* plane_cost -- iterface for 
   //                                 computing plane cost
   //
   // Out:
   // void
   ///////////////////////////////////////////////////////
   void ViewPropagation(const int& cur_iter,
     const IPlaneCost* plane_cost);

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
   void PlaneRefinement(const double& z_max, 
     const double& n_max, const double& z_thres,
     const IPlaneCost* plane_cost);

   void LeftRightCheck(int** valid);
   void FillInvalid(int** valid);
   void WeightedMedian(int** valid, 
     const int wnd_size, const double gamma);
   void PostProcessing();

   void PlaneToDisp();

   // random number generator
   RNG rng_;
   // color image
   Mat img_[kViewNum];
   // disparity image
   Mat dis_[kViewNum];
   // image property
   int wid_;
   int hei_;
   int max_dis_;
   int dis_scale_;
   // plane parameter
   Plane** plane_[kViewNum];
   // minimum plane cost
   double** min_cost_[kViewNum];
   // const paramters for plane refinement
   const double kMaxNorm_ = 1.0;
   const double kZStopThres_ = 0.1;
};

