// g++ -o main main.cpp -I. -I/usr/local/include/eigen3
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <igl/cat.h>

using namespace std;
using namespace igl;
using namespace Eigen;


template <class T>
void matlab_print(const string name, const T & X)
{
  cout<<name<<"=["<<endl<<X<<endl<<"];"<<endl;
}

int main(int argc, char * argv[])
{
  Eigen::MatrixXd A(3,4);
  A << 
    3,5,4,5,
    1,2,4,2,
    1,1,2,5;
  matlab_print("A",A);
  Eigen::MatrixXd B(3,4);
  B << 
    13,15,14,15,
    11,12,14,12,
    11,11,12,15;
  matlab_print("B",B);
  Eigen::MatrixXd C;
  C = cat(1,A,B);
  matlab_print("cat(1,A,B)",C);
  C = cat(2,A,B);
  matlab_print("cat(2,A,B)",C);
  
//   Eigen::MatrixXd R1(1,5);
//   Eigen::MatrixXd R2(1,5);
//   std::vector<MatrixXd> matbin;
//   Eigen::MatrixXd Ra;
//   R1 << 1,2,3,4,5;
//   R2 << 10,11,12,13,14;
//   matbin.push_back(R1);
//   matbin.push_back(R2);
//   cat(matbin, Ra);
//   std::cout << Ra << std::endl;
}
