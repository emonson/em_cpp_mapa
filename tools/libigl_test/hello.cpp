#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <iostream>
int main(int argc, char * argv[])
{
  if(argc>1)
  {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(argv[1],V,F);
    std::cout<<"Hello, mesh with "<<V.rows()<<" vertices!"<<std::endl;
  } else {
    std::cout<<"Hello, world!"<<std::endl;
  }
  return 0;
}
