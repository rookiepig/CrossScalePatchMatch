///////////////////////////////////////////////////////
// File: CommonFunc
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
#include<gflags\gflags.h>
using namespace std;
using namespace cv;

const int kViewNum = 2;
const double kDoubleEps = numeric_limits<double>::epsilon();
const double kDoubleMax = numeric_limits<double>::max();

enum RefView{ kLeft = 0, kRight = 1 };

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

// handle image border
inline int HandleBorder(const int& loc, const int& size) {
  if (loc < 0) {
    return loc + size;
  }
  if (loc >= size) {
    return loc - size;
  }
  return loc;
}