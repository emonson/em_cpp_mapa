#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include "kMeansRex.h"

/* IGL writeDMAT casts all matrix contents to double, and only the column and row
   counts are integers. Binary files have ascii 0 0\n on the first line, whereas ascii
   files list [num columns] [num rows]\n on the first line. Then both have a single
   linear sequence of doubles after that, either sizeof(double) for binary
   or "%lg\n" for each in ascii (0.17 precision)
*/

int main(int argc, char * argv[])
{
  Eigen::MatrixXd M;
  igl::readDMAT("artificial_data_rev1.dmat",M);
  unsigned int K = 3;
  
  KMeans::KMeansRex km(M, K);
  std::cout << M << std::endl;

  return 0;
}
