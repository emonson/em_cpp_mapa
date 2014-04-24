#include <Eigen/Core>
#include <igl/readDMAT.h>
#include <cstdio>
#include "kMeansRex.h"


#include <stdexcept>
#undef eigen_assert
#define eigen_assert(x) \
  if (!x) { throw (std::runtime_error("TEST FAILED: Some wrong values were generated")); }

#include <cmath>
#define close_enough(a,b) (std::abs(a-b) < 0.0001)
	

int main(int argc, char * argv[])
{
  // Read in test data
  Eigen::ArrayXXd X;
  std::cout << "Reading in Artifical 3D test data (rev1)" << std::endl;
  igl::readDMAT( "artificial_data_rev1.dmat", X );
  
	// ---------------------------
  // Actual KMeansRex object
  std::cout << "KMeansRex with mapa seeding – 3 clusters" << std::endl;
  
  KMeans::KMeansRex km(X, 3);
  
  // Results arrays
	ArrayXXd seeds = km.GetSeeds();
  ArrayXXd z = km.GetClusterAssignments();
	ArrayXXd centers = km.GetCenters();
  
	// ---------------------------
	// Print out results
	std::cout << "Seed points" << std::endl;
  std::cout << seeds << std::endl;
	
  std::cout << "Cluster assignments" << std::endl;
  for (int ii = 0; ii < z.size(); ii++)
  {
  	printf("%d ", (int)z(ii));
  	if (ii+1 % 40 == 0) printf("\n");
  }
  printf("\n");
	
	std::cout << "Cluster centers" << std::endl;
  std::cout << centers << std::endl;

	// ---------------------------
	// Test seed points values
	Eigen::MatrixXd Seeds_correct(3,3);
	Seeds_correct << -0.433417, -0.64096, 0.666714, 
	                 0.623972, 0.364006, -0.716225, 
	                 -0.762395, 0.527501, 0.385417;
	for (int ii = 0; ii < Seeds_correct.size(); ii++)
	{
		eigen_assert( close_enough( seeds(ii), Seeds_correct(ii) ) );
	}
	
	// Test first 10 cluster labels values
	Eigen::VectorXd Cluster_assigns_first10(10);
	Cluster_assigns_first10 << 2, 2, 1, 1, 2, 1, 1, 2, 2, 1;
	for (int ii = 0; ii < Cluster_assigns_first10.size(); ii++)
	{
		eigen_assert( close_enough( z(ii), Cluster_assigns_first10(ii) ) );
	}
	
	// Test last 10 cluster labels values
	Eigen::VectorXd Cluster_assigns_last10(10);
	Cluster_assigns_last10 << 2, 2, 2, 1, 2, 2, 2, 1, 2, 1;
	int ii_offset = z.size()-Cluster_assigns_last10.size();
	for (int ii = 0; ii < Cluster_assigns_last10.size(); ii++)
	{
		eigen_assert( close_enough( z(ii_offset+ii), Cluster_assigns_last10(ii) ) );
	}

	// Test cluster centroids values
	Eigen::MatrixXd Centers_correct(3,3);
	Centers_correct << 0.0355854, -0.580986, 0.248681, 
	                   0.329016, -0.129425, -0.207384,
	                  -0.270798, 0.252273, 0.153366;
	for (int ii = 0; ii < Centers_correct.size(); ii++)
	{
		eigen_assert( close_enough( centers(ii), Centers_correct(ii) ) );
	}
	
	// ---------------------------
	std::cout << "TEST PASSED" << std::endl;
	std::cout << "Seed points" << std::endl;
	// * * * Uncomment the following line to get the test to fail!! * * *
	// std::cout << "Seed points" << std::endl;
	std::cout << "Seed points" << std::endl;

	return 0;
}
