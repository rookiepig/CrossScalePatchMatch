///////////////////////////////////////////////////////
// File: Plane.h
// Desc: Simple plane class to record plane param for
//       for each pixel
//
// Author: rookiepig
// Date: 2014/04/01
///////////////////////////////////////////////////////
#pragma once
#include"commfunc.h"

class Plane {
 public:
   Plane() : norm_(0, 0, 0), point_(0, 0, 0), param_(0, 0, 0 ) {}
   Plane(const Vec3d& norm, const Point3d& point) :
     norm_(norm), point_(point) {
     update_param();
   }
   void set_point(const Point3d& point) {
     point_ = point;
   }
   void set_norm(const Point3d& norm) {
     norm_ = norm;
   }
   void update_param() {
     double denom = max(norm_[2], kDoubleEps);
     param_[0] = -norm_[0] / denom;
     param_[1] = -norm_[1] / denom;
     param_[2] = norm_.dot(point_) / denom;
   }
   Vec3d norm() const {
     return norm_;
   }
   Point3d point() const {
     return point_;
   }
   Vec3d param() const {
     return param_;
   }
 private:
   Vec3d norm_;
   Point3d point_;
   Vec3d param_;
};



