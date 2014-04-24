#ifndef AKERNELMAP_H
#define AKERNELMAP_H

#include "Geometry.h"
#include "Matrix.h"
#include "EuclideanMetric.h"
#include "Linalg.h"
#include "GaussianKernel.h"
#include "AnistropicGaussianKernel.h"

#include <stdlib.h>
#include <limits>
#include <vector>

template <typename TPrecision>
class AKernelMap{

  private:
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> X;
    DenseMatrix<TPrecision> fY;

    int knn;

    EuclideanMetric<TPrecision> l2metric;

    GaussianKernel<TPrecision> kernelX;
    GaussianKernel<TPrecision> *kernelY;
    GaussianKernel<TPrecision> kernelYtmp;
    
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;

    bool Xchanged;
    bool leaveout;

  public:
   
   //Create KernelMap for Y gvien an initial Mapping Xinit
   //use nn nearest neighbors to compute sigma for kernel inX
   //lout - leave one out 
   AKernelMap(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Xinit, 
       TPrecision alpha, int nn, bool lout): Y(Ydata), X(Xinit), knn(nn){
     leaveout = lout;
     kernelY = new GaussianKernel<TPrecision>[Y.N()];
     //compute kernelY valuse using anistropic kernels
     computeKY();
     fY = DenseMatrix<TPrecision>(X.M(), X.N()); 
     Xchanged = true;
     //compute f(Y)
     computefY();
     //compute kernel based on f(Y)
     computeKernelX(alpha);
   };

   
  
    
   //evalue objective function  = squared error
   TPrecision evaluate(){
     TPrecision e = 0;

     computefY();
     DenseVector<TPrecision> gfy(Y.M());
     for(int i=0; i < Y.N(); i++){
        g(i, gfy);
        e += l2metric.distanceSquared(Y, i, gfy); 
     }

     gfy.deallocate();

     return e;
   };
   

   //Gradient descent for all points 
   void gradDescent(int nIterations, TPrecision scaling, bool async,
       DenseMatrix<TPrecision> &cvData, TPrecision kCutY = -1, 
       TPrecision kCutX = 0, int nSamplesX = -1, int nSamplesY = -1){
     
     //gradient direction
     DenseVector<TPrecision> gx(X.M());

     //Vars for crossvalidation
     DenseMatrix<TPrecision> proj;
     DenseMatrix<TPrecision> rcst; 
     DenseVector<TPrecision> se;
     if(cvData.N() > 0 ){
       se = DenseVector<TPrecision>(cvData.N());
       proj = DenseMatrix<TPrecision> (X.M(), cvData.N());
       rcst = DenseMatrix<TPrecision> (Y.M(), cvData.N());
     }
     TPrecision mseCVprev = std::numeric_limits<TPrecision>::max( );

     //In case syncronous updates are used
     DenseMatrix<TPrecision> sync;
     if(!async){
      sync = DenseMatrix<TPrecision>(X.M(), X.N());
     }

     //do nIterations of gradient descent
     std::cout << "gradDescent all:" << std::endl;
     for(int i=0; i<nIterations; i++){

      //compute gradient fro each point
      //an update X
      for(int j=0; j < X.N(); j++){
        //compute gradient
        gradX(j, gx, kCutY, kCutX, nSamplesX, nSamplesY);
        //update if async updates
        if(async){
          Linalg<TPrecision>::ColumnAddScale(X, j, -scaling, gx);
          //mark Xchanged in order to update f(Y) if needed
          Xchanged = true;
        }
        //store gradient if syncronous updates
        else{
          for(int k=0; k<X.M(); k++){
            sync(k, j) = gx(k);
          }
        }
      }

      //Update if sync updates
      if(!async){
        Linalg<TPrecision>::AddScale(X, -scaling, sync, X);
        Xchanged = true;
      }


      
      std::cout << std::endl;
      std::cout << "----------------Iteration: " << i << std::endl;
      std::cout << "----------------MSE: " <<  evaluate()/Y.N() << std::endl;
      
      //crossvalidation reconstruction errors
      if(cvData.N() > 0 ){
        project(cvData, proj);
        unproject(proj, rcst);
         
        Precision mse = 0;
        Precision var = 0;
        for(int j=0; j<cvData.N(); j++){
          se(j) = l2metric.distanceSquared(rcst, j, cvData, j);
          mse += se(j);
        }
        mse /= cvData.N(); 
        for(int j=0; j<cvData.N(); j++){
          TPrecision tmp = se(j) -  mse;
          var += tmp*tmp;
        }
        var /= (cvData.N()-1);

        std::cout << "--------------CV MSE: " <<  mse << std::endl;
        std::cout << "--------------CV Var SE: " <<  var << std::endl;

        if(mseCVprev < mse){
          break;
        }

        mseCVprev = mse;
      }
      std::cout << std::endl << std::endl << std::endl;

     }

     //cleanup
     sync.deallocate();
     gx.deallocate();
     if(cvData.N() > 0 ){
      se.deallocate();
      proj.deallocate();
      rcst.deallocate();
     }
   };
  




   //f(y) - coordinate mapping
   void f( DenseVector<TPrecision> &y, DenseVector<TPrecision> &out, int lo = -1 ){
     Linalg<TPrecision>::Set(out, 0);


     //Compute anistropic gaussian kenrel for y
     DenseVector<int> knns(knn);
     DenseVector<TPrecision> knnDists(knn);
      
     Geometry<TPrecision>::computeKNN(Y, y, knns, knnDists, l2metric);
     kernelYtmp.setKernelParam(knnDists(knn-1)); 

     //mean.deallocate();
     
     //do kernel regression 
     DenseVector<TPrecision> k(Y.N());
     TPrecision sum = 0;
     int index = 0;
     for(int i=0; i<Y.N(); i++){
       if(index == lo){
         k(index) = 0;
       }
       else{
         k(index) = kernelYtmp.f(y, Y, i); 
         sum += k(index);
       }
       ++index;
     }

     for(int i=0; i<X.N(); i++){
       if(k(i)!=0){
         Linalg<TPrecision>::AddScale(out, k(i)/sum, X, i, out); 
       }
     }

     knns.deallocate();
     knnDists.deallocate();
     k.deallocate();  
   };




   //f(y_i) - coordinate mapping
   void f( int yi, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     for(int i=0; i<Y.N(); i++){
        if(KY(i, yi) != 0){
          Linalg<TPrecision>::AddScale(out, KYN(i, yi), X, i, out);
        }
     }
   };
  

  //g(y_index) - reconstruction mapping
  void g( int index, Vector<TPrecision> &out ){
     Linalg<TPrecision>::Set(out, 0);

     DenseVector<TPrecision> k(Y.N());
     TPrecision sum = 0;

     computefY();
     for(int i=0; i<Y.N(); i++){
         
       if(index == i && leaveout){
         k(i) = 0;
       }
       else{
         k(i) = kernelX.f(fY, index, fY, i);
       }

       if(k(i) != 0){
         sum += k(i);
       }
     }

     for(int i=0; i<Y.N(); i++){
       if( k(i) != 0){
        Linalg<TPrecision>::AddScale(out, k(i)/sum, Y, i, out); 
       } 
     } 

     k.deallocate();
   };


   //g(x) - reconstruction mapping
   void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     DenseVector<TPrecision> k(Y.N());
     TPrecision sum = 0;
     int index = 0;

     computefY();
     for(int i=0; i < Y.N(); i++){
         
       k(index) = kernelX.f(x, fY, i);
       if(k(index) != 0){
         sum += k(index);
       }
       ++index;
     }

     for(int i=0; i<Y.N(); i++){
       if( k(i) != 0){
        Linalg<TPrecision>::AddScale(out, k(i)/sum, Y, i, out); 
       } 
     } 

     k.deallocate();
   };

   

   //Compute gradient of f, e.g. \hat{x}_index 
   void gradX( int index, DenseVector<TPrecision> &gx, TPrecision kCutY, 
       TPrecision kCutX, int nSamplesX = -1, int nSamplesY = -1){
     
     Linalg<TPrecision>::Zero(gx);
     
     //g(f(y_index))
     DenseVector<TPrecision> gfy(Y.M());

     //kernel values & derivatives for K(f(y_index) - f(y_j));
     DenseVector<TPrecision> kx(X.N());
     DenseMatrix<TPrecision> kxd(X.M(), X.N());

     //sum of kernel values
     DenseVector<TPrecision> sumkxd(X.M());
     
     //gradient for each component of \hat{x}_index
     DenseMatrix<TPrecision> gradient(Y.M(), X.M());

     //Temp vars.
     DenseVector<TPrecision> gxtmp(X.M());
     DenseVector<TPrecision> diff(Y.M());
     TPrecision tmp;
     
     //update fY if necessary
     computefY();

     //Compute gradient
     int nX = nSamplesX;
     if(nX <= 0 ){
      nX = X.N();
     }
     int nY = nSamplesY;
     if(nY <= 0 ){
      nY = Y.N();
     }
     for(int is=0; is < nY; is++){

      int i = is;
      if( nSamplesY > 0 ){
        i = (int)( (rand()/(double)RAND_MAX) * (Y.N()-1));   
      }

      if(KYN(i, index) <= kCutY){ 
        continue;
      }
      
      //x-kernel values & derivatves at f(y)
      TPrecision sumkx = 0;
      sumkxd.zero();

      for(int j=0; j< X.N(); j++){
        
        //derivative and kernel value
        if(i == j && leaveout){
          kx(j) = 0;
        }
        else{
          kx(j) = kernelX.gradf(fY, i, fY, j, gxtmp);
        }

        //multipily kernel derivative by d[f(y_i)-f(y_j) / dx_index 
        //and store result in matrix kxd
        sumkx += kx(j);
        if(kx(j) == 0 || i == j){
          for(int k=0; k<X.M(); k++){
            kxd(k, j) = 0;
          }
        }
        else{
          TPrecision df = KYN(i, index) - KYN(j, index) ;
          for(int k=0; k<X.M(); k++){
            kxd(k, j) = gxtmp(k) * df;
            sumkxd(k) += kxd(k, j);
          }
        }
      }

      if(sumkx == 0){
        continue; 
      }

      TPrecision sumkx2 = sumkx*sumkx;

      //g(f(y_i)) 
      Linalg<TPrecision>::Zero(gfy);
      for(int j=0; j<Y.N(); j++){
        if( kx(j) != 0){
          Linalg<TPrecision>::AddScale(gfy, kx(j)/sumkx, Y, j, gfy); 
        } 
      }

     
      //Gradient matrix of (g \cric f)(f(y)) for each component of \hat{x}_index 
      Linalg<TPrecision>::Zero(gradient);
      for(int js=0; js<nX; js++){
        int j = js;
        if(nSamplesX > 0){
          j = (int)( (rand()/(double)RAND_MAX) * (X.N()-1));   
        }
        if(kx(j) > kCutX){
          //nk++;
          for(int n=0; n<X.M(); n++){
            tmp =  ( kxd(n, j) * sumkx - kx(j) * sumkxd(n) ) / sumkx2;
            for(int m=0; m<Y.M(); m++){
              gradient(m, n) +=  tmp * Y(m, j);
            }
          }
        }
      }

      //d E / d \hat{x}_index
      Linalg<TPrecision>::Subtract(gfy, Y, i, diff);
      Linalg<TPrecision>::Scale(diff, 2, diff);
      Linalg<TPrecision>::Multiply(gradient, diff, gxtmp, true);
      Linalg<TPrecision>::Add(gx, gxtmp, gx);

     }


     //cleanup
     gradient.deallocate();
     gxtmp.deallocate();
     diff.deallocate();
     kx.deallocate();
     kxd.deallocate();
     sumkxd.deallocate();
     gfy.deallocate();

   };



   //get original Y's
   DenseMatrix<TPrecision> getY(){
    return Y;
   };
   


   //get X (parameters for f
   DenseMatrix<TPrecision> getX(){
    return X;
   };


   //coordinate mapping fo Ypoints
   DenseMatrix<TPrecision> project(DenseMatrix<TPrecision> &Ypoints){

     DenseMatrix<TPrecision> proj(X.M(), Ypoints.N());
     project(Ypoints, proj);

     return proj;
   };

   //
   void project(DenseMatrix<TPrecision> &Ypoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> tmp(Y.M()); 
     DenseVector<TPrecision> xp(X.M()); 

     for(int i=0; i < Ypoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Ypoints, i, tmp);
       f(tmp, xp);
       Linalg<TPrecision>::SetColumn(proj, i, xp);
     }
     xp.deallocate();
     tmp.deallocate();
   };

  


   DenseMatrix<TPrecision> project(){
     return fY;
   };
  


   DenseMatrix<TPrecision> unproject(DenseMatrix<TPrecision> &Xpoints){
     DenseMatrix<TPrecision> proj(Y.M(), Xpoints.N());
     unproject(Xpoints, proj);     
     return proj;
   };
  


   void unproject(DenseMatrix<TPrecision> &Xpoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> tmp(X.M()); 
     DenseVector<TPrecision> yp(Y.M()); 
     for(int i=0; i < Xpoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Xpoints, i, tmp);
       g(tmp, yp);
       Linalg<TPrecision>::SetColumn(proj, i, yp);
     }
     yp.deallocate();
     tmp.deallocate();
   };


   TPrecision getSigmaX(){
     return kernelX.getKernelParam();
   };



private:


  void computefY(){
    if(Xchanged){
      std::cout << "updating f(Y)" << std::endl;
      Xchanged = false;
      DenseVector<TPrecision> tmp(X.M());
      for(int i=0; i<Y.N(); i++){
        f(i, tmp);
        Linalg<TPrecision>::SetColumn(fY, i, tmp);
      }
      tmp.deallocate();
    }
  };
 


  void computeKernelX(TPrecision alpha){
    DenseMatrix<int> knns(knn+1, X.N());
    DenseMatrix<TPrecision> knnDists(knn+1, X.N());
    Geometry<TPrecision>::computeKNN(fY, knns, knnDists, l2metric);
    
    TPrecision sigma = 0;
    for(int i=0; i < fY.N(); i++){
      sigma += knnDists(knn, i);
    }
    sigma/=fY.N();
    kernelX.setKernelParam(alpha*sigma);

    knns.deallocate();
    knnDists.deallocate();

  };




  void computeKY(){
    int N = Y.N();
    KY = DenseMatrix<TPrecision>(N, N);
    sumKY = DenseVector<TPrecision>(N);
    Linalg<TPrecision>::Set(sumKY, 0);
    KYN = DenseMatrix<TPrecision>(N, N);

    DenseMatrix<int> knns(knn+1, Y.N());
    DenseMatrix<TPrecision> knnDists(knn+1, Y.N());
    Geometry<TPrecision>::computeKNN(Y, knns, knnDists, l2metric);


    for(int i=0; i<N; i++){
      kernelY[i].setKernelParam(knnDists(knn, i)); 
    }


    std::cout << "Compute KY" << std::endl;
    for(int i=0; i < N; i++){
      for(int j=0; j < N; j++){
        if(j == i && leaveout){
          KY(j, i) = 0; 
        }
        else{
          KY(j, i) = kernelY[i].f(Y, j, Y, i); 
          sumKY(i) += KY(j, i);
        }
      }
    }

    for(int i=0; i < KY.M(); i++){
      for(int j=0; j< KY.N(); j++){
        KYN(i, j) = KY(i, j) / sumKY(j); 
      }
    }

    knns.deallocate();
    knnDists.deallocate();

  };

  
}; 


#endif
