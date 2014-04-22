#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <igl/writeDMAT.h>
using namespace igl;
#include <cstdio>

/* IGL writeDMAT casts all matrix contents to double, and only the column and row
   counts are integers. Binary files have ascii 0 0\n on the first line, whereas ascii
   files list [num columns] [num rows]\n on the first line. Then both have a single
   linear sequence of doubles after that, either sizeof(double) for binary
   or "%lg\n" for each in ascii (0.17 precision)
*/

int main(int argc, char * argv[])
{
  if(argc <= 2)
  {
    printf("USAGE:\n  ./example [input path] [output path]\n");
    return 1;
  }
  Eigen::MatrixXd M;
  readDMAT(argv[1],M);
  // ascii
  writeDMAT(argv[2], M, true);
  // binary
  // writeDMAT(argv[2], M, false);
  return 0;
}
