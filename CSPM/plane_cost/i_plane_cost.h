///////////////////////////////////////////////////////
// File: i_plane_cost.h
// Desc: interface for calculating plane cost
// 
// Author: rookiepig
// Date: 2014/04/02
//
///////////////////////////////////////////////////////
#pragma once
#include"../commfunc.h"
#include"../plane.h"

class IPlaneCost {
 public:
    IPlaneCost(void) {}
    virtual ~IPlaneCost(void) {}
    ///////////////////////////////////////////////////////
    // Func: GetPlaneCost
    // Desc: get aggregated cost with plane pramater
    // In:
    // const int& ref_x - reference pixel location
    // const int& ref_y - reference pixel location
    // const Plane& plane - plane paramter
    // const RefView& view - enum to decide current view
    // Out:
    // double - aggregated cost
    ///////////////////////////////////////////////////////
    virtual double GetPlaneCost(
      const int& ref_x,
      const int& ref_y,
      const Plane& plane,
      const RefView& view
     ) const = 0;
};
