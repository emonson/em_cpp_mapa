#ifndef KMMRegression_H
#define KMMRegression_H


#include "KMM2.h"
#include "Geometry.h"
#include "Matrix.h"
#include "EuclideanMetric.h"
#include "SquaredEuclideanMetric.h"
#include "KernelDensity.h"
#include "Linalg.h"
#include "LinalgIO.h"
#include "GaussianKernel.h"
#include "EpanechnikovKernel.h"
#include "Random.h"
#include "PCA.h"

#include <stdlib.h>
#include <limits>
#include <math.h>


template <typename TPrecision>
class KMMRegression : public KMM<TPrecision>{

  private:

    DenseMatrix<TPrecision> L;
    DenseMatrix<TPrecision> coeff;
    
    TPrecision lambda;


  public:
  
   virtual void cleanup(){ 
    KMM<TPrecision>::cleanup();     
    L.deallocate();
    coeff.deallocate();
   };

   //Create KernelMap 
   KMMRegression(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zinit,
       DenseMatrix<TPrecision> labels, TPrecision alpha, TPrecision lambdaa, 
       unsigned int nnSigma, unsigned int nnY, unsigned int nnX) :
       KMM<TPrecision>(Ydata, Zinit, alpha, nnSigma, nnY, nnX), 
       L(labels), lambda(lambdaa){
     
   };


 
   //evalue objective function, squared error
   virtual TPrecision mse(TPrecision &o){
     TPrecision e = KMM<TPrecision>::mse(o);
     TPrecision r = 0;
     
     //DenseVector<TPrecision> predict(L.M());

     for(unsigned int i=0; i < KMM<TPrecision>::Y.N(); i++){
       /*
       for(unsigned int j=0; j<L.M(); j++){
        predict(j) = coeff(0, j);
        for(unsigned int k=0; k<fY.M(); k++){
          predict(j) += fY(k, i) * coeff(k+1, j);
        }
       }
       r += sl2metric.distance(L, i, predict);
       */
       //r += leftoutLM(i);
       r += leftoutKR(i);

     }
     //predict.deallocate();

     std::cout << "r: " << r/KMM<TPrecision>::Y.N() << std::endl;
     std::cout << "e: " << e << std::endl;
     return e + lambda * r / KMM<TPrecision>::Y.N();
   };

  
   TPrecision leftoutKR(unsigned int index){
     DenseVector<TPrecision> predict(L.M());
     TPrecision sumw = 0;
     for(unsigned int i=0; i<KMM<TPrecision>::fY.N(); i++){
       if(i == index) continue;
       TPrecision w = KMM<TPrecision>::kernelX.f(KMM<TPrecision>::fY, i, KMM<TPrecision>::fY, index);
       sumw +=w;
       Linalg<TPrecision>::AddScale(predict, w, L, i, predict);
     }
     Linalg<TPrecision>::Scale(predict, 1.f/sumw, predict);

     TPrecision r = KMM<TPrecision>::sl2metric.distance(L, index, predict);
     predict.deallocate();

     return r;
   };

   
   TPrecision leftoutLM(unsigned int index){    
    //compute left out residual
    DenseMatrix<TPrecision> A(KMM<TPrecision>::fY.N()-1, KMM<TPrecision>::fY.M()+1);
    DenseMatrix<TPrecision> b(L.N()-1, L.M());
    int ai = 0;
    for(unsigned int i=0; i<KMM<TPrecision>::fY.N(); i++){
      if(index == i) continue;
      A(ai, 0) = 1;
      for(unsigned int j=1; j<A.N(); j++){
        A(ai, j) = KMM<TPrecision>::fY(j-1, i);
      }
      
      for(unsigned int j=0; j<b.N(); j++){
        b(ai, j) = L(j, i);
      }
      ai++;
    }

    DenseMatrix<TPrecision> C = Linalg<TPrecision>::LeastSquares(A, b);
    A.deallocate();
    b.deallocate();       
    
    DenseVector<TPrecision> predict(L.M());

    for(unsigned int j=0; j<L.M(); j++){
      predict(j) = C(0, j);
      for(unsigned int k=0; k<KMM<TPrecision>::fY.M(); k++){
        predict(j) += KMM<TPrecision>::fY(k, index) * C(k+1, j);
      }
    }
    C.deallocate();

    TPrecision r = KMM<TPrecision>::sl2metric.distance(L, index, predict);
    predict.deallocate();

    return r;
   }
   






   virtual TPrecision mse(int index){
     TPrecision e = KMM<TPrecision>::mse(index);
     TPrecision r = 0;
     
     
     //Temp vars.
     DenseVector<TPrecision> predict(L.M());

     for(unsigned int i=0; i < KMM<TPrecision>::knnY; i++){
       int nn = KMM<TPrecision>::KNNY(i, index);
       //regression residual
       for(unsigned int j=0; j<L.M(); j++){
         predict(j) = coeff(0, j);
         for(unsigned int k=0; k<KMM<TPrecision>::fY.M(); k++){
           predict(j) += KMM<TPrecision>::fY(k, nn) * coeff(k+1, j);
        }
       }
       r += KMM<TPrecision>::sl2metric.distance(L, nn, predict);
     }

     predict.deallocate();

     return e + lambda * r / KMM<TPrecision>::knnY;
   };
  

   
     
     


private:

  virtual void startIteration(int iter){
    KMM<TPrecision>::startIteration(iter);
    updateLM();
  }


  void updateLM(){      
    //update linear model
    DenseMatrix<TPrecision> A(KMM<TPrecision>::fY.N(), KMM<TPrecision>::fY.M()+1);
    for(unsigned int i=0; i<A.M(); i++){
      A(i, 0) = 1;
      for(unsigned int j=1; j<A.N(); j++){
        A(i, j) = KMM<TPrecision>::fY(j-1, i);
      }
    }
      
    DenseMatrix<TPrecision> b(L.N(), L.M());
    for(unsigned int i=0; i<b.M(); i++){
      for(unsigned int j=0; j<b.N(); j++){
        b(i, j) = L(j, i);
      }
    }

    coeff.deallocate();
    coeff = Linalg<TPrecision>::LeastSquares(A, b);
    A.deallocate();
    b.deallocate();
  }

}; 


#endif

