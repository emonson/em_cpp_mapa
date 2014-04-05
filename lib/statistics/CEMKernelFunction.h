#ifndef CEMKERNELFUNCTION_H
#define CEMKERNELFUNCTION_H

#include "Vector.h"
#include "Matrix.h"

template <typename TPrecision>
class CEMKernelFunction : public CEMFunction<TPrecision>{

private:
  //location of kernels in x and averaging positions in f(x)
  DenseMatrix<TPrecision> cX;
  DenseMatrix<TPrecision> cY;
  TPrecision s2;
  SquaredEuclideanMetric<TPrecision> l2;
  

public:

  CEMKernelFunction(DenseMatrix<TPrecision> &X, DenseMatrix<TPrecision> &Y, TPrecision sigma){
    cX = X;
    cY = Y;
    s2 = sigma*sigma; 
  };

  void setParameter(int pIndex, TPrecision &p){
    if(pIndex == 0){
      sigma =  p;
    } 
    else{
      cY.getData()[pIndex-1]  = p;
    }
  };
  
  void getParameter(int pIndex){
    if(pIndex == 0){
      return sigma;
    } 
    else{
      return cY.getData()[pIndex-1];
    } 
    
  };
  

  void evaluate(Vector<TPrecision> &x, Vector<TPrecision> &fx, Matrix<TPrecision> &gx) {
    double wsum = 0;
    double dwsum = 0;
    Linalg<TPrecision>::Zero(fx);
    bool gradient = gx.N() == x.N();

    for(int i=0; i<cX.N(); i++){
      TPrecision d = l2.distance(x, cX, i)
      TPrecision w = exp(-d/s2);
      Linalg<TPrecision>::AddScale(fx, cY, i, w, fx);

      if(gradient){
        for(int k=0; k<x.N(); k++){
          TPrecision dw = w*2*d[k];
          Linalg<TPrecision>::AddScaleColumn(gx, k, dw, cY, i)
          dwsum += dw;
        }
      } 
      
    }; 
    Linalg<TPrecision>::Scale(fx, 1.0/wsum, fx);
  
    if(gradient){
      Linalg<TPrecision>::Scale(fx, dwsum, fxs);
      for(int k=0; k<x.N(); k++){
        Linalg<TPrecision>::SubtractColumn(gx, k, fxs); 
      }
      Linalg<TPrecision>::Scale(gx, 1.0/wsum, gx);
    }
  };


};

#endif
