#ifndef EIGENRANDOMSVD_H
#define EIGENRANDOMSVD_H

#include "EigenRandomRange.h"
#include <Eigen/SVD>

namespace EigenLinalg{

  
  
class RandomSVD{

  public:
    Eigen::MatrixXd U;
    Eigen::VectorXd S;
    Eigen::VectorXd c;


    RandomSVD(Eigen::MatrixXd &Xin, int d, int nPowerIt = 0, bool
        center=false){
      
      using namespace Eigen;
      using namespace EigenLinalg;

      MatrixXd Q; 
      MatrixXd B;

      if(center){
        c = Xin.array().rowwise().sum() / Xin.cols();
        MatrixXd X = Xin.array().colwise() - c.array();
        
        Q = RandomRange::FindRange(X,d,nPowerIt);
        B = Q * X;
      }
      else{
        Q = RandomRange::FindRange(Xin,d,nPowerIt);
        B = Q * Xin;
      }

      JacobiSVD<MatrixXd> svd(B, ComputeThinU | ComputeThinV);
      S = svd.singularValues();
      U = Q * svd.matrixU();

    };



};


}


#endif 
