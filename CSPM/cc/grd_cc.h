#pragma once
#include "..\commfunc.h"
#include "..\cc_method.h"

// CVPR 11
#define BORDER_THRES 3
#define TAU_CLR 10.0
#define TAU_GRD 2.0
#define ALPHA 0.1

//
// TAD + GRD for Cost Computation
//
class GrdCC :
	public CCMethod
{
public:
	GrdCC(void)
	{
		printf( "\t\tGRD method for Cost Computation\n" );
	}
	~GrdCC(void) {}
public:
	void buildCV( const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol );
	void buildRightCV( const Mat& lImg, const Mat& rImg, const int maxDis, Mat* rCostVol );
};

