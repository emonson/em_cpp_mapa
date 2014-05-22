/* nextperms.cpp
 * Ver 0.2
 * Peter H. Li 2013 FreeBSD License 
 */
#include <algorithm>
#include "mex.h"

// Function declarations
mxArray *t_nextperms(const mxArray *inarr, const mwSize k);


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs != 2)
    mexErrMsgIdAndTxt("nextperms:nrhs", "Arguments should be 1) a vector and 2) the desired number of permutations.");
  
  if (!mxIsNumeric(prhs[0]))
    mexErrMsgIdAndTxt("nextperms:prhs", "First argument should be a numeric vector.");

  if (!mxIsNumeric(prhs[1]) || mxGetNumberOfDimensions(prhs[1]) != 2 || mxGetM(prhs[1]) != 1 || mxGetN(prhs[1]) != 1)
    mexErrMsgIdAndTxt("nextperms:prhs", "Second argument must be a nonnegative scalar number of permutations desired.");

  const double k0 = mxGetScalar(prhs[1]);
  if (k0 < 0) mexErrMsgIdAndTxt("nextperms:prhs", "Second argument must be nonnegative.");
  
  const mwSize k = k0;
  plhs[0] = t_nextperms(prhs[0], k);
}


/**
 * Populate output matrix with permutations
 */
template <typename T>
void nextperms(const T *in, const mwSize l, const mwSize k, T *out) {
  if (k == 0) return;
  
  // First col comes from in
  std::copy(in, in+l, out);
  std::next_permutation(out, out+l);

  // Remaining cols are copied over and permuted from previous col of out
  mwSize i;
  for (i = 1; i < k; ++i) {
    std::copy(out, out+l, out+l);
    out += l;
    std::next_permutation(out, out+l);
  }
}


/**
 * Just create the output array and get the data vectors in proper numeric 
 * type to pass along
 */
template <typename T> 
mxArray *run_nextperms(const mxArray *arr, const mwSize k) {
  const mwSize l = mxGetNumberOfElements(arr);
  mxArray *ret = mxCreateNumericMatrix(l, k, mxGetClassID(arr), mxREAL);
  const T* indata = static_cast<const T*>(mxGetData(arr));
  T* outdata = static_cast<T*>(mxGetData(ret));
  nextperms(indata, l, k, outdata); // Populates outdata in-place
  return ret;
}


/**
 * Choose the template to call based on Matlab input array numeric type
 */
mxArray *t_nextperms(const mxArray *inarr, const mwSize k) {
  mxArray *ret = NULL;
  
  switch (mxGetClassID(inarr)) {
    case mxDOUBLE_CLASS:
      ret = run_nextperms<double>(inarr, k);
      break;

    case mxSINGLE_CLASS:
      ret = run_nextperms<float>(inarr, k);
      break;

    case mxINT8_CLASS:
      ret = run_nextperms<signed char>(inarr, k);
      break;

    case mxUINT8_CLASS:
      ret = run_nextperms<unsigned char>(inarr, k);
      break;

    case mxINT16_CLASS:
      ret = run_nextperms<signed short>(inarr, k);
      break;

    case mxUINT16_CLASS:
      ret = run_nextperms<unsigned short>(inarr, k);
      break;

    case mxINT32_CLASS:
      ret = run_nextperms<signed int>(inarr, k);
      break;

    case mxUINT32_CLASS:
      ret = run_nextperms<unsigned int>(inarr, k);
      break;

    // Uncomment these if int64 is needed, but note that on some compilers
    // it's called "__int64" instead of "long long"
    //case mxINT64_CLASS:
      //ret = run_nextperms<signed long long>(inarr, k);
      //break;

    //case mxUINT64_CLASS:
      //ret = run_nextperms<unsigned long long>(inarr, k);
      //break;

    default:
      mexErrMsgIdAndTxt("Numerical:nextperms:prhs", "Unrecognized numeric array type.");
  }
  
  return ret;
}
