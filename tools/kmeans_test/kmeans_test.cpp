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
    igl::readDMAT( "/Users/emonson/Programming/em_cpp_mapa/data/artificial_data_rev1.dmat", X );

    // ---------------------------
    // Actual KMeansRex object
    std::cout << "KMeansRex with mapa seeding – 3 clusters" << std::endl;

    KMeans::KMeansRex km(X, 3);

    // Results arrays
    ArrayXXd seeds = km.GetSeeds();
    ArrayXi z = km.GetClusterAssignments();
    ArrayXXd centers = km.GetCenters();

    // ---------------------------
    // Print out results
    std::cout << "Seed points" << std::endl;
    std::cout << seeds << std::endl;

    std::cout << "Cluster assignments" << std::endl;
    std::cout << z.transpose() << std::endl;

    std::cout << "Cluster centers" << std::endl;
    std::cout << centers << std::endl;

    // ---------------------------
    // Test seed points values
    std::cout << "Testing Seed points" << std::endl;
    Eigen::ArrayXXd Seeds_correct(3,3);
    Seeds_correct << -0.433417, -0.64096, 0.666714,
            0.623972, 0.364006, -0.716225,
            -0.762395, 0.527501, 0.385417;
    for (int ii = 0; ii < Seeds_correct.size(); ii++)
    {
        eigen_assert( close_enough( seeds(ii), Seeds_correct(ii) ) );
    }

    // Test first 10 cluster labels values
    std::cout << "Testing cluster assignments" << std::endl;
    Eigen::ArrayXd Clusters_correct(z.size());
    Clusters_correct << 2, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 2,
            1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1,
            1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1,
            2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1,
            2, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1,
            1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2,
            2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1,
            2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1,
            2, 0, 0, 1, 1, 2, 0, 1, 0, 2, 2, 2, 0, 0, 2, 0, 0, 1, 2, 2, 2, 0, 0, 2, 0, 0,
            0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 0, 1, 2, 2, 1, 1, 0, 2, 0, 2, 0, 1, 2, 2,
            1, 1, 2, 2, 1, 2, 2, 0, 1, 1, 2, 2, 0, 2, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 0, 2,
            1, 0, 0, 2, 1, 2, 1, 2, 0, 1, 2, 1, 0, 1, 1, 1, 0, 2, 2, 2, 1, 1, 2, 2, 1, 1,
            0, 1, 0, 2, 1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 2, 2, 0, 2, 1,
            1, 1, 2, 2, 2, 0, 2, 0, 1, 2, 2, 1, 2, 1, 1, 0, 0, 2, 1, 2, 1, 1, 0, 2, 0, 1,
            1, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 0, 2,
            2, 1, 2, 2, 1, 2, 0, 0, 1, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2,
            1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1,
            2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1,
            1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2,
            1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1,
            2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2,
            2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1,
            2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1;
    for (int ii = 0; ii < Clusters_correct.size(); ii++)
    {
        eigen_assert( close_enough( z(ii), Clusters_correct(ii) ) );
    }

    // Test cluster centroids values
    std::cout << "Testing center points" << std::endl;
    Eigen::ArrayXXd Centers_correct(3,3);
    Centers_correct << 0.0355854, -0.580986, 0.248681,
            0.329016, -0.129425, -0.207384,
            -0.270798, 0.252273, 0.153366;
    for (int ii = 0; ii < Centers_correct.size(); ii++)
    {
        eigen_assert( close_enough( centers(ii), Centers_correct(ii) ) );
    }

    std::cout << "TESTS PASSED" << std::endl;

    return 0;
}
