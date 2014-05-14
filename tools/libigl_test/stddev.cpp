#include <Eigen/Core>

#include <iostream>

using namespace Eigen;

int main(int argc, char * argv[])
{
    ArrayXXd A = ArrayXXd::Random(10,3);
    std::cout << A << std::endl;
    
    // Column-wise
    ArrayXd colStdA = ((A.rowwise() - A.colwise().mean()).square().colwise().sum() / (double)(A.rows()-1)).sqrt();
    std::cout << std::endl << colStdA << std::endl;

    // Calculate standard deviation of the rows
    ArrayXd rowStdA = ((A.colwise() - A.rowwise().mean()).square().rowwise().sum() / (double)(A.cols()-1)).sqrt();
    std::cout << std::endl << rowStdA << std::endl;

  return 0;
}
