#ifndef MREGRESSION_H
#define MREGRESSION_H


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
class MRegression{

  private:    
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> Z;
    DenseMatrix<TPrecision> fY;
    DenseMatrix<TPrecision> L;
    DenseMatrix<TPrecision> R;
    DenseMatrix<TPrecision> *c;
    DenseMatrix<TPrecision> C;

    unsigned int knnSigma;
    
    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;

    
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;

    GaussianKernel<TPrecision> kernelY;
  
    TPrecision sX;
    bool Zchanged;
    bool verbose;


  public:
  
   void cleanup(){      
    R.deallocate();
    KY.deallocate();
    sumKY.deallocate();
    KYN.deallocate();
    Y.deallocate();
    Z.deallocate();
    fY.deallocate();
    L.deallocate();
   };

   //Manifold regression 
   MRegression(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zinit, 
       DenseMatrix<TPrecision> labels, unsigned int nnSigma ) :
       Y(Ydata), Z(Zinit), L(labels), knnSigma(nnSigma){
     
     verbose = true;//false;
     init();
   };


   TPrecision mse(){
     computefY();
     TPrecision r = 0;
     for(unsigned int i=0; i<Y.N(); i++){
	r += Linalg<TPrecision>::SquaredLengthColumn(R, i);
     }
     return r/Y.N();
   };



   //Gradient descent for all points 
   void gradDescent(unsigned int nIterations, TPrecision scaling){
     
     TPrecision objPrev = mse();
          
     if(verbose){
       std::cout << "Mse start: " << objPrev << std::endl;
     }


    
     
     //---Storage for syncronous updates 
     DenseMatrix<TPrecision> sync(Z.M(), Z.N());

     //---Do nIterations of gradient descent     
     DenseMatrix<TPrecision> Ztmp(Z.M(), Z.N());
     DenseMatrix<TPrecision> Zswap;

     //gradient direction
     DenseVector<TPrecision> gx(Z.M());     
     if(verbose){
      std::cout << "gradDescent all:" << std::endl;
     }

     for(unsigned int i=0; i<nIterations; i++){
      //compute gradient for each point
      TPrecision maxL = 0;
      for(unsigned int j=0; j < Z.N(); j++){
        //compute gradient
        //gradX(j, gx);
	 gradX(j, gx);
        
        //store gradient for syncronous updates
        TPrecision l = Linalg<TPrecision>::Length(gx);
	      if(maxL < l){
          maxL = l;
	      }
	      //Linalg<TPrecision>::Scale(gx, 1.f/l, gx);
        for(unsigned int k=0; k<Z.M(); k++){
          sync(k, j) = gx(k);
        }
      }
      std::cout << std::endl;


      //LinalgIO<TPrecision>::writeMatrix("G.data", sync);

      //std::cout << "maxL: " << maxL << std::endl;

      //sync updates
      TPrecision s;
      if(maxL == 0 )
	      s = -scaling;
      else{
	      s = -scaling * sX/maxL;
      }     
      if(verbose){
        std::cout << "scaling: " << s << std::endl;
      }
      
      
      //Approximate line search with quadratic fit
      DenseMatrix<TPrecision> A(3, 3);
      DenseMatrix<TPrecision> b(3, 1);
      Linalg<TPrecision>::Zero(A);

      b(0, 0) = mse();
      std::cout << b(0, 0) << std::endl;
      Linalg<TPrecision>::AddScale(Z, -1*s, sync, Ztmp);
      Zswap = Z;
      Z = Ztmp;
      Ztmp = Zswap;
      Zchanged = true;
      
      b(1, 0) = mse();
      std::cout << b(1, 0) << std::endl;
      Linalg<TPrecision>::AddScale(Zswap, -2*s, sync, Z);
      Zchanged = true;
      
      b(2, 0) = mse();
      std::cout << b(2, 0) << std::endl;
        
      A(0, 2) = 1;
      A(1, 0) = 1;
      A(1, 1) = -1;
      A(1, 2) = 1;
      A(2, 0) = 4;
      A(2, 1) = -2;
      A(2, 2) = 1;

      DenseMatrix<TPrecision> q = Linalg<TPrecision>::Solve(A, b);

      //do step
      if( q(0, 0) > 0){
        TPrecision h = -q(1, 0)/(2*q(0, 0));
        if(h < -2){
         h = -2;
        }
        else if( h > 1){
          h = 1;
        }
        Linalg<TPrecision>::AddScale(Ztmp, h*s, sync, Z);
      }
      else if( b(0,0) > b(1, 0) ){
        //do nothing step to -10*s
      }
      else{
        //small step
        Linalg<TPrecision>::AddScale(Ztmp, -s, sync, Z);
      }

      A.deallocate();
      b.deallocate();
      q.deallocate();
      
      //reconstruct(fY, gfY);
      //parametrize(gfY, fgfY);
      //Linalg<TPrecision>::Subtract(fgfY, fY, fgfY);
      //Linalg<TPrecision>::Subtract(Z,fgfY, Z);
      //Linalg<TPrecision>::Copy(fY, Z);
      Zchanged = true;



      std::cout << std::endl;
      TPrecision obj = mse(); 
      if(verbose){
        std::cout << "Iteration: " << i << std::endl;
        std::cout << "MSE: " <<  obj << std::endl;     
      }   
      if(objPrev < obj){
        break;
      }
      objPrev = obj;
     }


     //cleanup 
     sync.deallocate();
     gx.deallocate();
     Ztmp.deallocate();
   
   };
 

 


  void f(unsigned int index, Vector<TPrecision> &out, bool lo = true){
    Linalg<TPrecision>::Zero(out);
    TPrecision sumw = 0;
    for(unsigned int i=0; i < Z.N(); i++){
      if(lo && index == i) continue;
      TPrecision w = KY(i, index);
      Linalg<TPrecision>::AddScale(out, w, Z, i, out);
      sumw += w;
    }     
    Linalg<TPrecision>::Scale(out, 1.f/sumw, out);


  };




   void f( DenseVector<TPrecision> &y, Vector<TPrecision> &out){
     Linalg<TPrecision>::Zero(out);
     TPrecision sumw = 0;
     for(unsigned int i=0; i < Z.N(); i++){
       TPrecision w = kernelY.f(y, Y, i);
       Linalg<TPrecision>::AddScale(out, w, Z, i, out);
       sumw += w;
     }     
     Linalg<TPrecision>::Scale(out, 1.f/sumw, out);

   };






  


   //get original Y's
   DenseMatrix<TPrecision> getY(){
     return Y;
   };
   


   //get Z (parameters for f
   DenseMatrix<TPrecision> getZ(){
     return Z;
   };


   //coordinate mapping for Ypoints
   DenseMatrix<TPrecision> parametrize(DenseMatrix<TPrecision> &Ypoints){

     DenseMatrix<TPrecision> proj(Z.M(), Ypoints.N());
     parametrize(Ypoints, proj);

     return proj;
   };


   //
   void parametrize(DenseMatrix<TPrecision> &Ypoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> tmp(Y.M()); 
     DenseVector<TPrecision> xp(Z.M()); 

     for(unsigned int i=0; i < Ypoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Ypoints, i, tmp);
       f(tmp, xp);
       Linalg<TPrecision>::SetColumn(proj, i, xp);
     }
     xp.deallocate();
     tmp.deallocate();
   };

   //coordinate mapping for Ypoints
   DenseMatrix<TPrecision> predict(DenseMatrix<TPrecision> &Ypoints){

     DenseMatrix<TPrecision> pr(L.M(), Ypoints.N());
     predict(Ypoints, pr);

     return pr;
   };


   //
   void predict(DenseMatrix<TPrecision> &Ypoints, DenseMatrix<TPrecision> &pr){
     Zchanged = true;
     computefY(false);

     DenseVector<TPrecision> tmp(Y.M()); 
     DenseVector<TPrecision> xp(Z.M()); 
     DenseVector<TPrecision> predict(L.M());

     DenseVector<TPrecision> debug(Z.M());
     for(unsigned int i=0; i < Ypoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Ypoints, i, tmp);
       f(tmp, xp);
       if(i < Y.N()){
         f(i, debug);
        // std::cout << l2metric.distance(debug, xp) << " ";
       }
           

       for(unsigned int j=0; j<L.M(); j++){
          predict(j) = C(0, j);
          for(unsigned int k=0; k<fY.M(); k++){
            predict(j) += xp(k) * C(k+1, j);
          }
       }


       Linalg<TPrecision>::SetColumn(pr, i, predict);
     }
     xp.deallocate();
     tmp.deallocate();
     predict.deallocate();
   };


   DenseMatrix<TPrecision> &parametrize(){
     computefY();
     return fY;
   };
  

private:


   void init(){
     c = new DenseMatrix<TPrecision>[Y.N()];
     sX = 0;
     DenseMatrix<int> knn(knnSigma+1, Y.N());
     DenseMatrix<TPrecision> knnd(knnSigma+1, Y.N());
     Geometry<TPrecision>::computeKNN(Z, knn, knnd, sl2metric);
     for(unsigned int i=0; i<Y.N(); i++){
       sX += sqrt( knnd(knnSigma, i) ); 
     }
     sX /= Y.N();
     if(verbose){
       std::cout << "sX: " << sX << std::endl;
     }
     knn.deallocate();
     knnd.deallocate();


     if(verbose){
      std::cout << "Initalizing" << std::endl;
     }
    
     kernelY = GaussianKernel<TPrecision>( Z.M());
     
     if(knnSigma > Y.N()){
       knnSigma = Y.N();
     }

     if(verbose){
      std::cout << "Computing knn for Y" << std::endl;
     }


     if(verbose){
      std::cout << "Computing kernel values for Y" << std::endl;
     }

     computeKY();
     R = DenseMatrix<TPrecision>(L.M(), L.N());
     fY = Linalg<TPrecision>::Copy(Z);
     Zchanged = true;
     computefY(); 

     if(verbose) {   
      std::cout << "Initalizing done" << std::endl;
     }

   };






   void gradX(unsigned int index, DenseVector<TPrecision> gx){
     computefY();
     Linalg<TPrecision>::Zero(gx);
     
     for(unsigned int i=0; i<fY.N(); i++){
       if(i==index) continue;
       for(unsigned int j=0; j<fY.M(); j++){
         TPrecision tmp = 0;
         for(unsigned int k=0; k<L.M(); k++){
           tmp += R(k, i) * C(j+1, k);//c[i](j+1, k);
         }
         gx(j) -= KYN(i, index) * tmp;
         //gx(j) -= KYN(i, index) * R(j, i);
       }
     }
     
   };     





  void computefY(bool lo = true){
    if(Zchanged){
      Zchanged = false;
      DenseVector<TPrecision> tmp(Z.M());
      for(unsigned int i=0; i<Y.N(); i++){
        f(i, tmp, lo);
        Linalg<TPrecision>::SetColumn(fY, i, tmp);
        
        //Linalg<TPrecision>::Subtract(tmp, L, i, tmp);
        //Linalg<TPrecision>::SetColumn(R, i, tmp);
      }
      //for(unsigned int i=0;i<Y.N(); i++){
//	leftoutLM(i);
  //    }
      LM();

      tmp.deallocate();
    }
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

  void LM(){    
    //compute left out residual
    DenseMatrix<TPrecision> A(Y.N(), Z.M()+1);
    DenseMatrix<TPrecision> b(L.N(), L.M());
    for(unsigned int i=0; i<Y.N(); i++){
      A(i, 0) = 1;
      for(unsigned int j=1; j<A.N(); j++){
        A(i, j) = fY(j-1, i);
      }
      
      for(unsigned int j=0; j<b.N(); j++){
        b(i, j) = L(j, i);
      }
    }

    C.deallocate(); 
    C = Linalg<TPrecision>::LeastSquares(A, b);
    A.deallocate();
    b.deallocate();      
     
    
    DenseVector<TPrecision> predict(L.M());
    for(unsigned int index=0; index<Y.N(); index++){
      for(unsigned int j=0; j<L.M(); j++){
        predict(j) = C(0, j);
        for(unsigned int k=0; k<fY.M(); k++){
          predict(j) += fY(k, index) * C(k+1, j);
        }
      }
      Linalg<TPrecision>::Subtract(predict, L, index, predict);
      Linalg<TPrecision>::SetColumn(R, index, predict);
    }

    predict.deallocate();

   };

  /*void leftoutLM(unsigned int index){    
    //compute left out residual
    DenseMatrix<TPrecision> A(Y.N()-1, Z.M()+1);
    DenseMatrix<TPrecision> b(L.N()-1, L.M());
    int ai = 0;
    for(unsigned int i=0; i<Y.N(); i++){
      if(index == i) continue;
      A(ai, 0) = 1;
      for(unsigned int j=1; j<A.N(); j++){
        A(ai, j) = fY(j-1, i);
      }
      
      for(unsigned int j=0; j<b.N(); j++){
        b(ai, j) = L(j, i);
      }
      ai++;
    }

    c[index].deallocate(); 
    DenseMatrix<TPrecision> C = Linalg<TPrecision>::LeastSquares(A, b);
    c[index] = C;
    A.deallocate();
    b.deallocate();      
     
    
    DenseVector<TPrecision> predict(L.M());

    for(unsigned int j=0; j<L.M(); j++){
      predict(j) = C(0, j);
      for(unsigned int k=0; k<fY.M(); k++){
        predict(j) += fY(k, index) * C(k+1, j);
      }
    }

    Linalg<TPrecision>::Subtract(predict, L, index, predict);
    Linalg<TPrecision>::SetColumn(R, index, predict);
    predict.deallocate();

   };*/
 




  

  void computeKY(){
    unsigned int N = Y.N();
    KY = DenseMatrix<TPrecision>(N, N);
    sumKY = DenseVector<TPrecision>(N);
    Linalg<TPrecision>::Set(sumKY, 0);
    KYN = DenseMatrix<TPrecision>(N, N);


    Precision sigma = 0;
    DenseMatrix<int> knn(knnSigma+1, Y.N());
    DenseMatrix<TPrecision> knnd(knnSigma+1, Y.N());
    Geometry<TPrecision>::computeKNN(Y, knn, knnd, sl2metric);
    for(unsigned int i=0; i<Y.N(); i++){
      sigma += sqrt( knnd(knnSigma, i) ); 
    }
    sigma /= Y.N();
    if(verbose){
      std::cout << "sigmaY: " << sigma << std::endl;
    }
    kernelY = GaussianKernel<TPrecision>(sigma, Z.M());
    knn.deallocate();
    knnd.deallocate();

    if(verbose){
      std::cout << "Compute KY" << std::endl;
    }
    for(unsigned int i=0; i < N; i++){
      for(unsigned int j=0; j < N; j++){
        KY(j, i) = kernelY.f(Y, j, Y, i); 
        sumKY(i) += KY(j, i);
      }
    }

    for(unsigned int i=0; i < KY.M(); i++){
      for(unsigned int j=0; j< KY.N(); j++){
        KYN(i, j) = KY(i, j) / sumKY(j); 
      }
    }

  };

  
}; 


#endif

