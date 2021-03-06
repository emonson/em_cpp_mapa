/* KMeansSeedsRex.cpp
 Thin wrapper for a Matlab MEX function interface to KMeansRexCore.cpp

Contains:
  mexFunction : Map matlab arrays to Eigen arrays, and call the core library.

Dependencies:
  libkmeansrex.so

Author: Mike Hughes (www.michaelchughes.com)
Date:   2 April 2013
*/

#include <iostream>
#include <time.h>
#include "mex.h"
#include "Eigen/Dense"
#include "KMeansRexCore.h"

using namespace Eigen;
using namespace std;


/*  DEFINE Custom Type Names to make code more readable
    ExtMat :  2-dim matrix/array externally defined (e.g. in Matlab or Python)
*/
typedef Map<ArrayXXd> ExtMat;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	if (nrhs < 2 || nrhs > 5 ) {
		mexErrMsgTxt("Needs >=2 args -- Xdata (NxD), K (int), initname (str), seed (int)");
		return;
	}

  // ============================================== Process optional inputs

  char* initname;
  if (nrhs >= 3) {
    initname = mxArrayToString( prhs[2] );
  } else {
    initname = (char *) "mapa";
  }

	if (nrhs == 4) {
	  int SEED = (int) mxGetScalar( prhs[3] );
  	set_seed( SEED );
  } else {
    set_seed( (int) clock() );
	}
	
  // ============================================== Process required input K
	int K = (int) mxGetScalar(prhs[1]);

	double* X_IN = mxGetPr(prhs[0]);	
	int N = mxGetM( prhs[0] );
	int D = mxGetN( prhs[0] );
	ExtMat X (X_IN, N, D);

  // =============================================  Input Sanity Checks
	if (K <= 1 ) {
	  mexErrMsgTxt("Need at least two clusters to avoid degeneracy.");
	  return;
	}
	if (K >= N) {
	  mexErrMsgTxt("Cannot request more clusters than data points.");
		return;
	}
	
  // =============================================  Allocate output Mu
	plhs[0] = mxCreateDoubleMatrix(K, D, mxREAL);
  ExtMat Mu ( mxGetPr(plhs[0]), K, D );
  
  // =============================================  Do the Work!
  // Initialize and Run Lloyd Alg to convergence!  
  init_Mu( X, Mu, initname);
}
