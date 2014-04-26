#include <Eigen/Core>
#include <cstdio>
#include "UtilityCalcs.h"
#include "igl/unique.h"

int main(int argc, char * argv[])
{
  int runs = 20;
  
  ArrayXi entries_unique(runs);
  bool sorted = false;
  
  for (int ii=0; ii < runs; ii++ )
  {
  	ArrayXi rs;
  	
  	// Switching to returning sorted results half-way through
  	if (ii == (int)runs/2)
  	{ 
  		sorted = !sorted;
  		std::cout << "--" << std::endl;
  	}
		
		// Generate random sample
  	rs = MAPA::UtilityCalcs::RandSample(20, 10, sorted);
  	
  	// Check that sample has been done without substitution
  	// TODO: Should probably check whether gives error for too large or small values...
  	MatrixXi rs2 = rs.matrix();
  	MatrixXi un_rs;
  	VectorXi IA, IC;
  	igl::unique_rows(rs2, un_rs, IA, IC);
  	
  	entries_unique(ii) = (int)(rs.size() == un_rs.size());
  	
    std::cout << rs.transpose() << " - " << entries_unique(ii) << std::endl;
	}
	
	if ((entries_unique > 0).all())
	{
		std::cout << "TEST PASSED" << std::endl;
	}
	else
	{
		std::cout << "FAILED" << std::endl;
	}
	
  return 0;
}
