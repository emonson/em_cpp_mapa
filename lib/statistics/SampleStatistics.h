#ifndef SAMPLESTATISTICS_H
#define SAMPLESTATISTICS_H


#include "Linalg.h"

template <typename TPrecision>
class SampleStatistics {

  public:

    static DenseMatrix<TPrecision> Covariance(DenseMatrix<TPrecision> data){
        DenseMatrix<TPrecision> cov = Linalg<TPrecision>::Multiply(data, data, false, true);
        TPrecision *dptr = cov.data();
        for(int i=0; i<cov.N()*cov.M(); i++){
          dptr[i] /= (TPrecision)(data.N()-1);
        }
        return cov;
    };
    

    
    static DenseMatrix<TPrecision> Covariance(DenseMatrix<TPrecision> data,
        DenseVector<TPrecision> mean){

        DenseMatrix<TPrecision> centered(data.M(), data.N());
        Linalg<TPrecision>::SubtractColumnwise(data, mean, centered);
        DenseMatrix<TPrecision> cov = Linalg<TPrecision>::Multiply(centered, centered, false, true);
        TPrecision *dptr = cov.data();
        for(int i=0; i<cov.M()*cov.N(); i++){
          dptr[i] /= (TPrecision)(data.N()-1);
        }
        centered.deallocate();
        return cov;
    };
    
    static DenseMatrix<TPrecision> Covariance(DenseMatrix<TPrecision> data,
        DenseVector<TPrecision> mean, DenseVector<TPrecision> weights){

        DenseMatrix<TPrecision> centered(data.M(), data.N());
        Linalg<TPrecision>::SubtractColumnwise(data, mean, centered);

        TPrecision sum = 0;
        for(int i=0; i<centered.N(); i++){
          Linalg<TPrecision>::ScaleColumn(centered, i, sqrt(weights(i)));
          sum +=  weights(i);
        }

        DenseMatrix<TPrecision> cov = Linalg<TPrecision>::Multiply(centered, centered, false, true);
        TPrecision *dptr = cov.data();
        for(int i=0; i<cov.M()*cov.N(); i++){
          dptr[i] /= sum;
        }
        centered.deallocate();
        return cov;
    };

   

    static DenseVector<TPrecision> Mean(DenseMatrix<TPrecision> &data){
        DenseVector<TPrecision> tmp(data.N());
        Linalg<TPrecision>::Set(tmp, 1.f/data.N());
        DenseVector<TPrecision> m = Linalg<TPrecision>::Multiply(data, tmp);
        tmp.deallocate();
        return m;
    };
   


    static DenseVector<TPrecision> Mean(DenseMatrix<TPrecision> &data,
      DenseVector<TPrecision> weights){
        DenseVector<TPrecision> m = Linalg<TPrecision>::Multiply(data, weights);
        TPrecision sumw = 0;
        for(int i=0; i<weights.N(); i++){
          sumw+=weights(i);
        }
        Linalg<TPrecision>::Scale(m, 1.0/sumw, m);
        return m;
    };


    static TPrecision Variance(DenseVector<TPrecision> x){
      static TPrecision dummy;
      return Variance(x, dummy);
    }

    static TPrecision Variance(DenseVector<TPrecision> x, TPrecision &m){
      m = Linalg<TPrecision>::Sum(x) /x.N(); 
      Linalg<TPrecision>::Subtract(x, m, x);
      return Linalg<TPrecision>::SquaredLength(x) / (x.N()-1); 
    };
};

#endif
