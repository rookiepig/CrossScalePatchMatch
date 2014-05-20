///////////////////////////////////////////////////////
// File: commfunc.cc
// Desc: implementation of global functions
//
// Author: Zhang Kang
// Date: 2013/09/06
///////////////////////////////////////////////////////
#include"commfunc.h"

// constant time median filter
void MedianFilter(cv::InputArray iImage_, cv::OutputArray oImage_, int r) {
  cv::Mat iImage = iImage_.getMat();
  cv::Size imageSize = iImage.size();
  CV_Assert(iImage.depth() == CV_8U);

  cv::Mat tmp(imageSize, iImage.type());
  ctmf(iImage.data, tmp.data, imageSize.width, imageSize.height,
    iImage.step1(), tmp.step1(), r,
    iImage.channels(), imageSize.area() * iImage.channels());

  if (oImage_.getMat().size() != imageSize || oImage_.getMat().depth() != CV_8U || oImage_.getMat().type() != CV_8UC1) {
    oImage_.create(imageSize, iImage.type());
  }
  tmp.copyTo(oImage_.getMat());
}