#ifndef CONDITIONALGAUSSIANDENSITY_H
#define CONDITIONALGAUSSIANDENSITY_H

#include "Kernel.h"
#include "SampleStatistics.h"
#include "Linalg.h"

#define PI 3.14159265358979323846

template <typename TPrecision>
class ConditionalGaussianDensity {

  private:
    TPrecision u1;
    DenseVector<TPrecision> u2;
    DenseVector<TPrecision> tmp;
    DenseVector<TPrecision> rCoef;
    TPrecision var;
    TPrecision n;
    int index;

    

  public:
    
    ConditionalGaussianDensity(){
    };
    
    ~ConditionalGaussianDensity(){
      u2.deallocate();
      tmp.deallocate();
      rCoef.deallocate();
    };
    

    TPrecision p(DenseVector<TPrecision> &x){
        TPrecision d = x(index) - E(x);
        d*=d;
        return n * exp(-d/(2*var)); 
    };

    TPrecision E(DenseVector<TPrecision> &x){
      int jn = 0;
      for(int j=0; j<x.N(); j++){
        if(j!=index){
          tmp(jn) = x(j) - u2(jn);
          jn++; 
        }
      }
      
      return u1 - Linalg<TPrecision>::Dot(tmp, rCoef);
    };

    TPrecision variance(){
      return var;
    };


    void estimateConditionalDensity(DenseMatrix<TPrecision> data, int i){
      
      DenseVector<TPrecision> mean =
         SampleStatistics<TPrecision>::Mean(data);
      estimate(data, mean, i);
      mean.deallocate();
    }

    void estimateConditionalDensity(DenseMatrix<TPrecision> data,
        DenseVector<TPrecision> weights, int i){
      
      DenseVector<TPrecision> mean =
         SampleStatistics<TPrecision>::Mean(data, weights);
      estimate(data, mean, i);
      mean.deallocate();
    }


  private:

    void estimate(DenseMatrix<TPrecision> data,
       DenseVector<TPrecision> mean, int ind){
       u2.deallocate();
       tmp.deallocate();
       rCoef.deallocate();

       index = ind;
       u2 = DenseVector<TPrecision>(mean.N()-1);
       rCoef = DenseVector<TPrecision>(mean.N()-1);
       tmp = DenseVector<TPrecision>(mean.N()-1);

       DenseMatrix<TPrecision> cov =
         SampleStatistics<TPrecision>::Covariance(data, mean);

       TPrecision cov11 = cov(index, index);
       DenseVector<TPrecision> cov12(cov.N()-1);
       DenseMatrix<TPrecision> cov21(cov.N()-1, 1);
       DenseMatrix<TPrecision> cov22(cov.N()-1, cov.N()-1);
       u1 = mean(index);

       int in = 0;
       for(int i=0; i<cov.N(); i++){
         if(i!=index){
           int jn = 0;
           for(int j=0; j<cov.N(); j++){
             if(j!=index){
               cov22(in, jn) = cov(i, j);
               jn++; 
             }
           }
           cov12(in) = cov(index, i);
           cov21(in, 0) = cov(i, index);
           u2(in) = mean(i);
           in++;
         }
       }

       int info = 0;
       TPrecision rcond = 0;
       DenseMatrix<TPrecision> rCoefTmp = Linalg<TPrecision>::SolveSPD(cov22,
           cov21, rcond, info);

       if(info > 0 ){
         //std::cout << rcond << std::endl;
         Linalg<TPrecision>::Set(rCoef, 0);
       }
       else{
        Linalg<TPrecision>::ExtractColumn(rCoefTmp, 0, rCoef);
       }
       //for(int i=0; i<rCoef.N(); i++){
       //  std::cout << rCoef(i) << " ";
       //}
       //std::cout << std::endl;

       var = cov11 - Linalg<TPrecision>::Dot(rCoef, cov12);
       if(var <= 0){
         var = 0.00000001;
       }
       n = 1/sqrt(2.0*PI*var);

       cov.deallocate();
       cov12.deallocate();
       cov22.deallocate();
       cov21.deallocate();
       rCoefTmp.deallocate();
    };

   

};

#endif
