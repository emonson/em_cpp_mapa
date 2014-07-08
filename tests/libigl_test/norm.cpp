#include <Eigen/Dense>

#include <iostream>

using namespace std;

int main(int argc, char * argv[])
{
  // Dense
  // Eigen select
  Eigen::MatrixXf m(2,2);
  m << 1, 2, 3, 4;
  // select equivalent of loop over members(i), (m(i) < 3) ? 0 : m(i)
  cout << m << endl;
  cout << (m.array() < 3).select(0,m) << endl;
  cout << (m.array() < 3) << endl;
  cout << m.rowwise().squaredNorm() << endl;

  return 0;
}
