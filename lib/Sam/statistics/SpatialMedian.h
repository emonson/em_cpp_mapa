#ifndef SPATIALMEDIAN_H
#define SPATIALMEDIAN_H


#include "Linalg.h"

template <typename TPrecision>
class SpatialMedain {

  public:

    //Weighted spatial median through weiszfelds algorithm
    static DenseVector<TPrecision> Weiszfeld(DenseMatrix<TPrecision> X, DenseVector<TPrecision> w, int maxIter = 100){
    
       DenseMatrix<TPrecision> W(1, w.N(), w.data());
       Linalg<TPRecision>::Scale(W, 1.0/Linalg<TPrecision>::Sum(w), W);
       
       //intialize to mean
       DenseVector<TPrecision> y1 = Linalg<TPrecision>::Multiply(X, W);
       DenseVector<TPrecision> y2(y1.N());
       DenseVector<TPrecision> ySwap;

       for(int k=0; k<maxIter; k++){
          TPrecision sumw = 0; 
          Linalg<TPrecision>::Zero(y2);
          for(int i=0; i<X.N(); i++){
            TPrecision s = w(i) / l2.distance(X, i, y1) ;
            sumw += s;
            Linalg<TPrecision>::AddScale(y2, X, i, s, y2);
          }
          Linalg<TPrecision>::Scale(y2, 1.0/sumw, y2);
          ySwap = y1;
          y1 = y2;
          y2 = ySwap; 
       };
    };
};
    

#endif
