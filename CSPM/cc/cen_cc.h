#pragma once
#include "..\commfunc.h"
#include "..\cc_method.h"

#define CENCUS_WND 9
#define CENCUS_BIT 80

//
// Cencus for Cost Computation
//
class CenCC :
	public CCMethod
{
public:
	CenCC(void)
	{
		printf( "\t\tCencus for Cost Computation\n" );
	}
	~CenCC(void) {}
public:
	void buildCV( const Mat& lImg, const Mat& rImg, const int maxDis, Mat* costVol );
	void buildRightCV( const Mat& lImg, const Mat& rImg, const int maxDis, Mat* rCostVol );
};

