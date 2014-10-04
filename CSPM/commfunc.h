///////////////////////////////////////////////////////
// File: commfunc.h
// Desc: global functions, constants and headers
//
// Author: Zhang Kang
// Date: 2013/09/06
///////////////////////////////////////////////////////
#pragma  once
#include<opencv2/opencv.hpp>
#include<string>
#include<iostream>
#include<iomanip>
#include<algorithm>
#include<limits>
#include<cstdlib>
#include<gflags/gflags.h>
#include<omp.h>
#include<bitset>
// C Header
#include"ctmf.h"
using namespace std;
using namespace cv;

const int kViewNum = 2;
// not use too small eps to avoid overflow
const double kDoubleEps = 0.00000001;
const double kDoubleMax = numeric_limits<double>::max();

enum RefView{ kLeft = 0, kRight = 1 };

#pragma comment(lib, "ShLwapi.lib")

//
// gflags lib
//
#ifdef _DEBUG
#pragma comment(lib, "gflags_debug.lib")
#else
#pragma comment(lib, "gflags.lib")
#endif


//
// Opencv Lib 2.4.6
//
#ifdef _DEBUG
#pragma comment( lib, "opencv_calib3d248d.lib" )
#pragma comment( lib, "opencv_contrib248d.lib" )
#pragma comment( lib, "opencv_core248d.lib" )
#pragma comment( lib, "opencv_features2d248d.lib" )
#pragma comment( lib, "opencv_flann248d.lib" )
#pragma comment( lib, "opencv_gpu248d.lib" )
#pragma comment( lib, "opencv_highgui248d.lib" )
#pragma comment( lib, "opencv_imgproc248d.lib" )
#pragma comment( lib, "opencv_legacy248d.lib" )
#pragma comment( lib, "opencv_ml248d.lib" )
#pragma comment( lib, "opencv_nonfree248d.lib" )
#pragma comment( lib, "opencv_objdetect248d.lib" )
#pragma comment( lib, "opencv_photo248d.lib" )
#pragma comment( lib, "opencv_stitching248d.lib" )
#pragma comment( lib, "opencv_superres248d.lib" )
#pragma comment( lib, "opencv_ts248d.lib" )
#pragma comment( lib, "opencv_video248d.lib" )
#pragma comment( lib, "opencv_videostab248d.lib" )
#else
#pragma comment( lib, "opencv_calib3d248.lib" )
#pragma comment( lib, "opencv_contrib248.lib" )
#pragma comment( lib, "opencv_core248.lib" )
#pragma comment( lib, "opencv_features2d248.lib" )
#pragma comment( lib, "opencv_flann248.lib" )
#pragma comment( lib, "opencv_gpu248.lib" )
#pragma comment( lib, "opencv_highgui248.lib" )
#pragma comment( lib, "opencv_imgproc248.lib" )
#pragma comment( lib, "opencv_legacy248.lib" )
#pragma comment( lib, "opencv_ml248.lib" )
#pragma comment( lib, "opencv_nonfree248.lib" )
#pragma comment( lib, "opencv_objdetect248.lib" )
#pragma comment( lib, "opencv_photo248.lib" )
#pragma comment( lib, "opencv_stitching248.lib" )
#pragma comment( lib, "opencv_superres248.lib" )
#pragma comment( lib, "opencv_ts248.lib" )
#pragma comment( lib, "opencv_video248.lib" )
#pragma comment( lib, "opencv_videostab248.lib" )
#endif

//
// Global Functions
//

// output matrix
template<class T>
void PrintMat(const Mat& mat)
{
	int rows = mat.rows;
	int cols = mat.cols;
	printf("\n%d x %d Matrix\n", rows, cols);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			cout << mat.at<T>(r, c) << "\t";
		}
		printf("\n");
	}
	printf("\n");
}

// fast float floor to int
inline int Floor2Int(double d) {
  // minus -0.5 to floor
  const double dme = .5f - 1.5e-8;
  d -= dme;
  // magic number
  d = d + 6755399441055744.0;
  return ((int*)&d)[0];    // 0 for little endian, 1 for big endian
}

// fast float rount to int
inline int Round2Int(double d) {
  // magic number
  d = d + 6755399441055744.0;
  return ((int*)&d)[0];    // 0 for little endian, 1 for big endian
}


inline double FastFabs(double x) {
  int tmp = (int&)x & 0x7FFFFFFFFFFFFFFF;
  return (double&)tmp;
}
// handle image border
inline int HandleBorder(const int& loc, const int& size) {
  //if (loc < 0 || loc >= size) {
  //  // mod too slow !!!
  //  return ( loc + size ) % size;
  //}
  if (loc < 0) {
    // CV_Assert(loc + size >= 0);
    return loc + size;
    // return 0;
  }
  if (loc >= size) {
    // CV_Assert(loc - size < size);
    return loc - size;
    // return size - 1;
  }
  return loc;
}
// handle image border macro
//#define HANDLE_BORDER(loc, size) \
//  ((loc) < 0 ? (loc)+(size) : \
//  ((loc) >= (size) ? (loc)-(size) : (loc)))
//

// constant time median filter
//inline void MedianFilter(cv::InputArray iImage_, cv::OutputArray oImage_, int r) {
//  cv::Mat iImage = iImage_.getMat();
//  cv::Size imageSize = iImage.size();
//  CV_Assert(iImage.depth() == CV_8U);
//
//  cv::Mat tmp(imageSize, iImage.type());
//  ctmf(iImage.data, tmp.data, imageSize.width, imageSize.height,
//    iImage.step1(), tmp.step1(), r,
//    iImage.channels(), imageSize.area() * iImage.channels());
//
//  if (oImage_.getMat().size() != imageSize || oImage_.getMat().depth() != CV_8U || oImage_.getMat().type() != CV_8UC1) {
//    oImage_.create(imageSize, iImage.type());
//  }
//  tmp.copyTo(oImage_.getMat());
//}

// #define MY_DEBUG
#define USE_OMP