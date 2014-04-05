#ifndef EIGENRANDOMRANGE_H
#define EIGENRANDOMRANGE_H

#include "Random.h"
#include <Eigen/QR>

namespace EigenLinalg{

class RandomRange{

  public:


    static Eigen::MatrixXd FindRange(Eigen::MatrixXd X, int
        d, int nPowerIt = 0){

      using namespace Eigen;


      static Random<double> rand;
      
      MatrixXd N(X.cols(), d);
      for(unsigned int i=0; i< N.rows(); i++){
        for(unsigned int j=0; j< N.cols(); j++){
          N(i, j) = rand.Normal();
        }
      }

      
      MatrixXd Q = X *N;
      HouseholderQR<MatrixXd> qr(Q);

      if(nPowerIt > 0){
        MatrixXd Z;
        for( int i=0; i<nPowerIt; i++){
          Z = X.transpose() * qr.householderQ();
          qr.compute(Z);

          Q = X * Z;
          qr.compute(Q);
        }
      }
      
      return qr.householderQ();
    };

};

}

#endif 
