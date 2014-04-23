#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include "kMeansRex.h"

typedef Map<ArrayXXd> ExtMat;

#include <stdexcept>
#undef eigen_assert
#define eigen_assert(x) \
  if (!x) { throw (std::runtime_error("Some wrong values were generated")); }

#include <cmath>
#define close_enough(a,b) (std::abs(a-b) < 0.0001)
	

int main(int argc, char * argv[])
{
  Eigen::MatrixXd M;
  igl::readDMAT("artificial_data_rev1.dmat",M);
  
  // std::cout << M << std::endl;

  KMeans::KMeansRex km;
  
  unsigned int K = 3;
  int N = M.rows();
  int D = M.cols();
  double Mu_OUT[K*D];
  char method[5] = "mapa";
  
  ExtMat X  ( M.data(), N, D);
  ExtMat Mu ( Mu_OUT, K, D);
  double Z_OUT[N];
  ExtMat Z  ( Z_OUT, N, 1);
	

  // Generate seed points
  km.init_Mu( X, Mu, method);
	
	std::cout << "Seed points" << std::endl;
  std::cout << Mu << std::endl;
	
	// Test Mu values
	Eigen::MatrixXd Seeds_correct(3,3);
	Seeds_correct << -0.433417, -0.64096, 0.666714, \
	               0.623972, 0.364006, -0.716225, \
                -0.762395, 0.527501, 0.385417;
	for (int ii = 0; ii < Seeds_correct.size(); ii++)
	{
		eigen_assert( close_enough( Mu(ii), Seeds_correct(ii) ) );
	}
	
	// Run clustering
  km.run_lloyd( X, Mu, Z, 100 );
  
  std::cout << "Cluster assignments" << std::endl;
  std::cout << Z.transpose() << std::endl;
	
	// Test Z values
	Eigen::VectorXd Cluster_assigns_first10(10);
	Cluster_assigns_first10 << 2, 2, 1, 1, 2, 1, 1, 2, 2, 1;
	for (int ii = 0; ii < Cluster_assigns_first10.size(); ii++)
	{
		eigen_assert( close_enough( Z(ii), Cluster_assigns_first10(ii) ) );
	}
	
	// Test Z values
	Eigen::VectorXd Cluster_assigns_last10(10);
	Cluster_assigns_last10 << 2, 2, 2, 1, 2, 2, 2, 1, 2, 1;
	int jj = Z.size()-Cluster_assigns_last10.size();
	for (int ii = 0; ii < Cluster_assigns_last10.size(); ii++)
	{
		eigen_assert( close_enough( Z(jj+ii), Cluster_assigns_last10(ii) ) );
	}

	std::cout << "Cluster centers" << std::endl;
  std::cout << Mu << std::endl;

	// Test Mu values
	Eigen::MatrixXd Centers_correct(3,3);
	Centers_correct << 0.0355854, -0.580986, 0.248681, \
	               0.329016, -0.129425, -0.207384, \
                -0.270798, 0.252273, 0.153366;
	for (int ii = 0; ii < Centers_correct.size(); ii++)
	{
		eigen_assert( close_enough( Mu(ii), Centers_correct(ii) ) );
	}
	
  return 0;
}
