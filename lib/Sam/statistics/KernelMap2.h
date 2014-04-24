#ifndef KERNELMAP2_H
#define KERNELMAP2_H

#include "Matrix.h"
#include "EuclideanMetric.h"
#include "Linalg.h"
#include "Kernel.h"

#include <stdlib.h>
#include <limits>
#include <vector>

template <typename TPrecision>
class KernelMap2{
  public:
   typedef typename std::vector< DenseVector<TPrecision> > VList;
   typedef typename VList::iterator VListIterator;


  private:
    //VList Ysamples;
    VList Y;
    VList X;
    VList fY;

    int dimensionX;
    int dimensionY;

    EuclideanMetric<TPrecision> l2metric;

    Kernel<TPrecision, TPrecision> &kernelX;
    Kernel<TPrecision, TPrecision> &kernelY;
    
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;

    int nky;
    bool Xchanged;

  public:
   
   KernelMap(Kernel<TPrecision, TPrecision> &ky, Kernel<TPrecision, TPrecision>
       &kx, int dimX, int dimY, int N): kernelX(kx), kernelY(ky), fY(N){

     KY = DenseMatrix<TPrecision>(N, N);
     sumKY = DenseVector<TPrecision>(N);
     KYN = DenseMatrix<TPrecision>(N, N);
     Linalg<TPrecision>::Set(sumKY, 0);
     nky = 0;
     
     dimensionX = dimX;
     dimensionY = dimY;

     for(int i=0; i<N; i++){
      fY[i] = DenseVector<TPrecision>(dimensionX);
     }

     Xchanged = true;
   };

   
   int add(DenseVector<TPrecision> y, DenseVector<TPrecision> x){
     X.push_back(x);
     Y.push_back(y);
     //DenseVector<TPrecision> ys(y);
     //Ysamples.push_back(ys);

     Xchanged = true;
     
     return X.size() - 1;
   };


   int add(DenseVector<TPrecision> y){
     DenseVector<TPrecision> x(dimensionX);
     f(y, x);
     
     X.push_back(x);
     Y.push_back(y);
     
     //DenseVector<TPrecision> ys(y);
     //Ysamples.push_back(ys);

     Xchanged = true;
     
     return X.size() - 1;
   }  
    

   TPrecision evaluate(bool leaveout){
     TPrecision e = 0;

     computefY(leaveout);
     DenseVector<TPrecision> gfy(dimensionY);
     for(int i=0; i < Y.size(); i++){
        if(leaveout){
          g(fY[i], gfy, i);
        }
        else{
          g(fY[i], gfy);
        }
        e += l2metric.distanceSquared(Y[i], gfy); 
     }

     gfy.deallocate();

     return e;
   };
   

   //Gradient descent for all points 
   void gradDescent(int nIterations, TPrecision scaling, bool async,
       DenseMatrix<TPrecision> &cvData, bool leaveout = false, TPrecision kCutY
       =-1, TPrecision kCutX = 0, int nSamplesX = -1, int nSamplesY = -1){
     DenseVector<TPrecision> gx(dimensionX);


     DenseMatrix<TPrecision> proj;
     DenseMatrix<TPrecision> rcst; 
     DenseVector<TPrecision> se;
     if(cvData.N() > 0 ){
       se = DenseVector<TPrecision>(cvData.N());
       proj = DenseMatrix<TPrecision> (dimensionX, cvData.N());
       rcst = DenseMatrix<TPrecision> (dimensionY, cvData.N());
     }
     TPrecision mseCVprev = std::numeric_limits<TPrecision>::max( );

     DenseMatrix<TPrecision> sync;
     if(!async){
      sync = DenseMatrix<TPrecision>(dimensionX, X.size());
     }

     std::cout << "gradDescent all:" << std::endl;
     for(int i=0; i<nIterations; i++){

      for(int j=0; j < X.size(); j++){
        gradX(j, gx, leaveout, kCutY, kCutX, nSamplesX, nSamplesY);
        if(async){
          DenseVector<TPrecision> tmp = X[j];
          Linalg<TPrecision>::AddScale(tmp, -scaling, gx, tmp);
          Xchanged = true;
        }
        else{
          for(int k=0; k<dimensionX; k++){
            sync(k, j) = gx(k);
          }
        }
      }

      if(!async){
        for(int j=0; j < sync.N(); j++){
          DenseVector<TPrecision> tmp = X[j];
          for(int k=0; k<dimensionX;k++){
            tmp(k) += -scaling * sync(k, j);
          }
        }
        Xchanged = true;
      }

      std::cout << std::endl;
      std::cout << "----------------Iteration: " << i << std::endl;
      std::cout << "----------------MSE: " <<  evaluate(leaveout)/Y.size() << std::endl;
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

     sync.deallocate();
     gx.deallocate();
     if(cvData.N() > 0 ){
      se.deallocate();
      proj.deallocate();
      rcst.deallocate();
     }
   };
  

   //Gradient descent for a single point
   void gradDescent(int nIterations, TPrecision scaling, int index, bool
       leaveout = false, TPrecision kCutY =-1, TPrecision kCutX = 0){
     DenseVector<TPrecision> gx(dimensionX);

     std::cout << "gradDescent: " << index << std::endl;
     std::cout << "E: " << evaluate(leaveout) << std::endl;
     for(int i=0; i<nIterations; i++){
      gradX(index, gx, leaveout, kCutY, kCutX);
      
      DenseVector<TPrecision> &tmp = X[index];
      Linalg<TPrecision>::AddScale(tmp, -scaling, gx, tmp);
      Xchanged = true;

      std::cout << "Iteration: " << i << ", ";
      std::cout << "E: " << evaluate(leaveout) << std::endl;
     }

     gx.deallocate();
   };



   
   void f( Vector<TPrecision> &y, Vector<TPrecision> &out, int lo = -1 ){
     Linalg<TPrecision>::Set(out, 0);

     computeKY(lo>=0);

     DenseVector<TPrecision> k(Y.size());
     TPrecision sum = 0;
     int index = 0;
     for(VListIterator it = Y.begin(); it != Y.end(); ++it){
       if(index == lo){
         k(index) = 0;
       }
       else{
         k(index) = kernelY.f(*it, y); 
         sum += k(index);
       }
       ++index;
     }

     for(int i=0; i<Y.size(); i++){
       if(k(i)!=0){
       //  std::cout << k(i) << ", "; 
        Linalg<TPrecision>::AddScale(out, k(i)/sum, X[i], out); 
       }
     };
     //std::cout << std::endl;

     k.deallocate();
     
   };



   void f( int yi, Vector<TPrecision> &out, bool lo){
     Linalg<TPrecision>::Set(out, 0);

     computeKY(lo);
     TPrecision sum = sumKY(yi);
     for(int i=0; i<Y.size(); i++){
        if(KY(i, yi) != 0){
          Linalg<TPrecision>::AddScale(out, KY(i, yi)/sum, X[i], out);
        }
     }
   };
  



   void g( Vector<TPrecision> &x, Vector<TPrecision> &out, int lo = -1){
     Linalg<TPrecision>::Set(out, 0);

     DenseVector<TPrecision> k(Y.size());
     TPrecision sum = 0;
     int index = 0;

     computefY( lo>=0 );
     for(VListIterator it = fY.begin(); it != fY.end(); ++it){
         
       if(index == lo){
         k(index) = 0;
       }
       else{
         k(index) = kernelX.f(*it, x);
       }

       if(k(index) != 0){
         sum += k(index);
       }
       ++index;
     }

     for(int i=0; i<Y.size(); i++){
       if( k(i) != 0){
        //std::cout << "i: " << i << "-" << k(i)/sum << ", "; 
        Linalg<TPrecision>::AddScale(out, k(i)/sum, Y[i], out); 
       } 
     } 
     //std::cout << std::endl << std::endl;


     k.deallocate();
   };

   

   //Compute gradient of f, e.g. \hat{x}_index 
   void gradX( int index, DenseVector<TPrecision> &gx, bool leaveout, TPrecision
       kCutY, TPrecision kCutX, int nSamplesX = -1, int nSamplesY = -1){
     Linalg<TPrecision>::Zero(gx);
     
     //g(f(y_index))
     DenseVector<TPrecision> gfy(dimensionY);

     //kernel values & derivatives for K(f(y_index) - f(y_j));
     DenseVector<TPrecision> kx(X.size());
     DenseMatrix<TPrecision> kxd(dimensionX, X.size());
     //sum of kernel values
     DenseVector<TPrecision> sumkxd(dimensionX);
     
     //gradient for each component of \hat{x}_index
     DenseMatrix<TPrecision> gradient(dimensionY, dimensionX);

     //Temp vars.
     DenseVector<TPrecision> gxtmp(dimensionX);
     DenseVector<TPrecision> diff(dimensionY);
     TPrecision tmp;
     
     //Update Y kernel values if necessary
     computeKY(leaveout);
     //update fY if necessary
     computefY(leaveout);

     //debug
     /*
     int ni = 0;
     int nk = 0;
     int niZero = 0;
     */

     //Compute gradient
     int nX = nSamplesX;
     if(nX <= 0 ){
      nX = X.size();
     }
     int nY = nSamplesY;
     if(nY <= 0 ){
      nY = Y.size();
     }
     for(int is=0; is < nY; is++){

      int i = is;
      if( nSamplesY > 0 ){
        i = (int)( (rand()/(double)RAND_MAX) * (Y.size()-1));   
      }

      if(KYN(i, index) <= kCutY){ 
        //niZero++;
        continue;
      }
      
      //x-kernel values & derivatves at f(y)
      TPrecision sumkx = 0;
      sumkxd.zero();

      for(int j=0; j< X.size(); j++){
        
        //derivative and kernel value
        if(i == j && leaveout){
          kx(j) = 0;
        }
        else{
          kx(j) = kernelX.gradf(fY[i], fY[j], gxtmp);
        }

        //mulipily kernel derivative by d[f(y_i)-f(y_j) / dx_index 
        //and store result in matrix kxd
        sumkx += kx(j);
        if(kx(j) == 0 || i == j){
          for(int k=0; k<dimensionX; k++){
            kxd(k, j) = 0;
          }
        }
        else{
          TPrecision df = KYN(i, index) - KYN(j, index) ;
          for(int k=0; k<dimensionX; k++){
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
      Linalg<TPrecision>::Set(gfy, 0);
      for(int j=0; j<Y.size(); j++){
        if( kx(j) != 0){
          Linalg<TPrecision>::AddScale(gfy, kx(j)/sumkx, Y[j], gfy); 
        } 
      }

     
      //Gradient matrix of (g \cric f)(f(y)) for each component of \hat{x}_index 
      Linalg<TPrecision>::Zero(gradient);
      for(int js=0; js<nX; js++){
        int j = js;
        if(nSamplesX > 0){
          j = (int)( (rand()/(double)RAND_MAX) * (X.size()-1));   
        }
        if(kx(j) > kCutX){
          //nk++;
          for(int n=0; n<dimensionX; n++){
            tmp =  ( kxd(n,j) * sumkx - kx(j) * sumkxd(n) ) / sumkx2;
            for(int m=0; m<dimensionY; m++){
              gradient(m, n) +=  tmp * Y[j](m);
            }
          }
        }
      }

      //d E / d \hat{x}_index
      Linalg<TPrecision>::Subtract(gfy, Y[i], diff);
      Linalg<TPrecision>::Scale(diff, 2, diff);
      Linalg<TPrecision>::Multiply(gradient, diff, gxtmp, true);
      Linalg<TPrecision>::Add(gx, gxtmp, gx);
     }


     gradient.deallocate();
     gxtmp.deallocate();
     diff.deallocate();
     kx.deallocate();
     kxd.deallocate();
     sumkxd.deallocate();
     gfy.deallocate();

/*
     std::cout << "uncut: " << ni ;
     std::cout << ", nonzero Y: " << Y.size() - niZero;
     std::cout << ", nonzero X: " << nk/((double)ni) << " | ";
*/
   };


   DenseMatrix<TPrecision> getY(){
    DenseMatrix<TPrecision> Ytmp(dimensionY, Y.size());
    int index = 0;
    for(VListIterator yit = Y.begin(); yit != Y.end(); ++yit){
      for(int k=0; k < dimensionY; k++){
        Ytmp(k, index) = (*yit)(k);
      }
      ++index;
    }

    return Ytmp;
   };
   
  
/*
   DenseMatrix<TPrecision> getYsamples(){
    DenseMatrix<TPrecision> Ytmp(dimensionY, Ysamples.size());
    int index = 0;
    for(VListIterator yit = Ysamples.begin(); yit != Ysamples.end(); ++yit){
      for(int k=0; k < dimensionY; k++){
        Ytmp(k, index) = (*yit)(k);
      }
      ++index;

    }

    return Ytmp;
   }

*/


   DenseMatrix<TPrecision> getX(){
    DenseMatrix<TPrecision> Xtmp(dimensionX, X.size());
    int index = 0;
    for(VListIterator xit = X.begin(); xit != X.end(); ++xit){
        for(int k=0; k < dimensionX; k++){
          Xtmp(k, index) = (*xit)(k);
        }
        ++index;
      }

    return Xtmp;
   };



   //
   DenseMatrix<TPrecision> project(DenseMatrix<TPrecision> &Ypoints){

     DenseMatrix<TPrecision> proj(dimensionX, Ypoints.N());
     project(Ypoints, proj);

     return proj;
   };




   void project(DenseMatrix<TPrecision> &Ypoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> tmp(dimensionY); 
     DenseVector<TPrecision> xp(dimensionX); 

     for(int i=0; i < Ypoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Ypoints, i, tmp);
       f(tmp, xp);
       Linalg<TPrecision>::SetColumn(proj, i, xp);
     }
     xp.deallocate();
     tmp.deallocate();
   };

  


   DenseMatrix<TPrecision> project(){

     DenseVector<TPrecision> xp(dimensionX); 

     DenseMatrix<TPrecision> proj(dimensionX, Y.size());
     for(int i=0; i < Y.size(); i++){
       f(i, xp, false);
       Linalg<TPrecision>::SetColumn(proj, i, xp);
     }
     xp.deallocate();
     return proj;
   };
  


   DenseMatrix<TPrecision> unproject(DenseMatrix<TPrecision> &Xpoints){
     DenseMatrix<TPrecision> proj(dimensionY, Xpoints.N());
     unproject(Xpoints, proj);     
     return proj;
   };
   



   void unproject(DenseMatrix<TPrecision> &Xpoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> tmp(dimensionX); 
     DenseVector<TPrecision> yp(dimensionY); 
     for(int i=0; i < Xpoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Xpoints, i, tmp);
       g(tmp, yp);
       Linalg<TPrecision>::SetColumn(proj, i, yp);
     }
     yp.deallocate();
     tmp.deallocate();
   };


private:

  void computeKernelX(){
    DenseMatrix<int> knns(knn+1, X.N());
    DenseMatrix<TPrecision> knnDists(knn+1, X.N());
    Geometry<TPrecision>::computeKNN(fY, knns, knnDists, l2metric);
    
    TPrecision sigma = 0;
    for(int i=0; i < fY.N(); i++){
      sigma += knnDists(knn, i);
    }
    sigma/=fY.N();
    kernelX = GaussianKernel<TPrecision>(3*sigma);

    knns.deallocate();
    knnDists.deallocate();
  };
  
  void computeKernelY(){
    DenseMatrix<int> knns(knn+1, Y.N());
    DenseMatrix<TPrecision> knnDists(knn+1, Y.N());
    Geometry<TPrecision>::computeKNN(Y, knns, knnDists, l2metric);
    
    TPrecision sigma = 0;
    for(int i=0; i < fY.N(); i++){
      sigma += knnDists(knn, i);
    }
    sigma/=fY.N();
    kernelY = GaussianKernel<TPrecision>(sigma);

    knns.deallocate();
    knnDists.deallocate();
  };




  void computefY(bool leaveout){
    if(Xchanged){
      std::cout << "updating f(Y)" << std::endl;
      Xchanged = false;
      for(int i=0; i<Y.size(); i++){
        f(i, fY[i], leaveout);
      }
    }
  };



  void computeKY(bool leaveout){
    if(nky == Y.size()){
      return;
    }

    std::cout << "Compute KY" << std::endl;
    for(int i=nky; i < Y.size(); i++){
      for(int j=0; j<Y.size(); j++){
        if(j == i && leaveout){
          KY(j, i) = 0; 
        }
        else{
          KY(j, i) = kernelY.f(Y[j], Y[i]); 
          sumKY(i) += KY(j, i);
        }
      }
    }
    for(int i=0; i < nky; i++){
      for(int j=nky; j<Y.size(); j++){ 
        if(j == i && leaveout){
          KY(j, i) = 0; 
        }
        else{
          KY(j, i) = kernelY.f(Y[j], Y[i]); 
          sumKY(i) += KY(j, i);
        }
      }
    }

    nky = Y.size();  

    //TODO: smarter update
    for(int i=0; i < nky; i++){
      for(int j=0; j< nky; j++){
        KYN(i, j) = KY(i, j) / sumKY(j); 
      }
    } 

  };

  
}; 


#endif
