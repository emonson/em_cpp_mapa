#ifndef FIRSTORDERKERNELREGRESSION_H
#define FIRSTORDERKERNELREGRESSION_H

#include <math.h>

#include "SquaredEuclideanMetric.h"
#include "Geometry.h"
#include "GaussianKernel.h"
#include "DenseVector.h"
#include "DenseMatrix.h"
#include "Linalg.h"

template<typename TPrecision>
class FirstOrderKernelRegression{
      
  public:
    FirstOrderKernelRegression(DenseMatrix<TPrecision> &data, DenseMatrix<TPrecision>
        &labels, GaussianKernel<TPrecision> &k, int knn):Y(data), X(labels), kernel(k){
	if(knn > X.N()){ knn = X.N(); }
	A = DenseMatrix<TPrecision>(knn, 1+X.M());
     	b = DenseMatrix<TPrecision>(knn, Y.M());

    };


    void cleanup(){
      A.deallocate();
      b.deallocate();
    };     


     void project(DenseVector<TPrecision> &x, DenseVector<TPrecision> &y, 
                  DenseVector<TPrecision> &out){
       DenseMatrix<TPrecision> sol = ls(x);
       for(unsigned int i=0; i<Y.M(); i++){
           out(i) = sol(0, i);
       }
       for(unsigned int j=0; j< X.M(); j++){
         TPrecision dprod = 0;
         for(unsigned int i=0; i<Y.M(); i++){
           dprod +=  (y(i)-sol(0, i)) * sol(j+1, i);
         }
         TPrecision length = Linalg<TPrecision>::LengthRow(sol, j+1); 
         dprod /= (length * length);
         for(unsigned int i=0; i<Y.M(); i++){
           out(i) += dprod * sol(j+1, i);
         }
       }
       sol.deallocate();
      
     };
     

     void evaluate(DenseVector<TPrecision> &x, DenseVector<TPrecision> &out, TPrecision *sse=NULL){
       DenseMatrix<TPrecision> sol = ls(x, sse);
       for(unsigned int i=0; i<Y.M(); i++){
         out(i) = sol(0, i);
       }
       sol.deallocate();
     };
 
 
      void evaluate( DenseVector<TPrecision> &x, Vector<TPrecision> &out,
Matrix<TPrecision> &J, double *sse=NULL){
        DenseMatrix<TPrecision> sol = ls(x, sse);
        for(unsigned int i=0; i<Y.M(); i++){
          out(i) = sol(0, i);
        }     
        for(unsigned int i=0; i< J.N(); i++){
          for(unsigned int j=0; j< J.M(); j++){
            J(j, i) = sol(1+i, j);
          }
        }
        sol.deallocate();
      };


  private:
    SquaredEuclideanMetric<TPrecision> sl2metric;
 
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> X;

    GaussianKernel<TPrecision> &kernel;

    DenseMatrix<TPrecision> A;
    DenseMatrix<TPrecision> b;



    DenseMatrix<TPrecision> ls(DenseVector<TPrecision> &x, TPrecision *sse=NULL){
      DenseVector<int> knn(A.M());
      DenseVector<TPrecision> knnDist(A.M());
      Geometry<TPrecision>::computeKNN(X, x, knn, knnDist, sl2metric);

     TPrecision wsum = 0; 
     for(unsigned int i=0; i < A.M(); i++){
       unsigned int nn = knn(i);
       TPrecision w = kernel.f(knnDist(i));
       A(i, 0) = w;
       for(unsigned int j=0; j< X.M(); j++){
         A(i, j+1) = (X(j, nn)-x(j)) * w;
       }

       for(unsigned int m = 0; m<Y.M(); m++){
	 b(i, m) = Y(m, nn) *w;
       }
       wsum += w*w;
     }
    
     DenseMatrix<TPrecision> sol = Linalg<TPrecision>::LeastSquares(A, b, sse);
     if(sse != NULL){ 
       for(int i=0; i<sol.N(); i++){
         sse[i] /= wsum;
       }
     }

     knn.deallocate();
     knnDist.deallocate();
     return sol;
   };


};


#endif

