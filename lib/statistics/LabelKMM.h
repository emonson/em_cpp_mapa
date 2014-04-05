#ifndef LabelKMM_H
#define LabelKMM_H

#include "Geometry.h"
#include "Matrix.h"
#include "EuclideanMetric.h"
#include "SquaredEuclideanMetric.h"
#include "KernelDensity.h"
#include "Linalg.h"
#include "GaussianKernel.h"

#include <stdlib.h>
#include <limits>

template <typename TPrecision>
class LabelKMM{

  private:
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> L;
    DenseMatrix<TPrecision> Z;
    DenseMatrix<TPrecision> fY;

    unsigned int knnSigma;
    unsigned int knnY;
    unsigned int knnX;

    unsigned int knnXUpdate;


    DenseMatrix<int> KNNY;
    DenseMatrix<TPrecision> KNNYD;


    //DenseMatrix<int> KNNX;
    //DenseMatrix<TPrecision> KNNXD;

    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;

    GaussianKernel<TPrecision> kernelX;
    GaussianKernel<TPrecision> *kernelY;
    GaussianKernel<TPrecision> kernelYtmp; 
    GaussianKernel<TPrecision> kernelL;

    
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;

    bool Zchanged;
    bool adaptive;
    
    TPrecision lWeight;

    TPrecision sX;
    TPrecision sX3s;     
    
    //
    DenseVector<TPrecision> wTmp;




  public:
  
   void cleanup(){      
    wTmp.deallocate();
    KNNY.deallocate();
    KNNYD.deallocate();
    KY.deallocate();
    sumKY.deallocate();
    KYN.deallocate();
    Y.deallocate();
    Z.deallocate();
    fY.deallocate();
    delete[] kernelY; 
   };

   //Create KernelMap 
   LabelKMM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> YLabel, DenseMatrix<TPrecision> Zinit, 
       TPrecision alpha, unsigned int nnSigma, TPrecision labelSigma, TPrecision
       labelWeight,  unsigned int nnY, unsigned int nnX, bool adapt = false) :
       Y(Ydata), Z(Zinit), knnSigma(nnSigma), knnY(nnY), knnX(nnX),
       adaptive(adapt), lWeight(labelWeight){
     
     kernelL.setKernelParam(labelSigma);
     init();
     computeKernelX(alpha);
   };



   void init(){
     wTmp = DenseVector<TPrecision>(Z.N());

     kernelYtmp = GaussianKernel<TPrecision>((int) Z.M());
     kernelX = GaussianKernel<TPrecision>((int) Z.M());
     sX = 1;
     sX3s = std::numeric_limits<TPrecision>::max();
     std::cout << "Initalizing" << std::endl;
     kernelY = new GaussianKernel<TPrecision>[Y.N()];
    
     if(knnY > Y.N()){
      knnY = Y.N();
     }
     if(knnX > Z.N()){ 
       knnX = Z.N();
     }
     if(knnSigma > Y.N()){
       knnSigma = Y.N();
     }

     std::cout << "Computing knn for Y" << std::endl;
     int knn = std::max(knnSigma, knnY);
     KNNY =  DenseMatrix<int>(knn, Y.N());
     KNNYD = DenseMatrix<TPrecision>(knn, Y.N());
     Geometry<TPrecision>::computeKNN(Y, KNNY, KNNYD, l2metric);

     knnXUpdate = 1;

     knn = std::max(knnSigma, knnX);
     //KNNX = DenseMatrix<int>(knn, Y.N());
     //KNNXD = DenseMatrix<TPrecision>(knn, Y.N());

     std::cout << "Computing kernel values for Y" << std::endl;
     computeKY();

     fY = DenseMatrix<TPrecision>(Z.M(), Z.N()); 
     Zchanged = true;
     computefY();

     //computeKNNX();
     std::cout << "Initalizing done" << std::endl;
   };     
   

   //evalue objective function, squared error
   TPrecision evaluate(){
     TPrecision e = 0;

     computefY();
     DenseVector<TPrecision> gfy(Y.M());
     for(unsigned int i=0; i < Y.N(); i++){
        g(i, gfy);
        e += l2metric.distanceSquared(Y, i, gfy); 
     }

     gfy.deallocate();

     return e;
   };
   

   //Gradient descent for all points 
   void gradDescent(unsigned int nIterations, TPrecision scaling, 
       DenseMatrix<TPrecision> &cvData, bool async = false){
     


     //---Initalize vars for crossvalidation
     DenseMatrix<TPrecision> proj;
     DenseMatrix<TPrecision> rcst; 
     DenseVector<TPrecision> se;
     if(cvData.N() > 0 ){
       se = DenseVector<TPrecision>(cvData.N());
       proj = DenseMatrix<TPrecision> (Z.M(), cvData.N());
       rcst = DenseMatrix<TPrecision> (Y.M(), cvData.N());
     }
     
     TPrecision mseCVprev = std::numeric_limits<TPrecision>::max( );
     TPrecision varCVprev = mseCVprev; 
     
     //---Storage for syncronous updates are used
     DenseMatrix<TPrecision> sync;
     if(!async){
      sync = DenseMatrix<TPrecision>(Z.M(), Z.N());
     }


     //---Do nIterations of gradient descent     
     
     //gradient direction
     DenseVector<TPrecision> gx(Z.M());
     std::cout << "gradDescent all:" << std::endl;
     for(unsigned int i=0; i<nIterations; i++){

      //update neareast neighbors of fY
      //if( (i+1) % knnXUpdate == 0){
      //  computeKNNX();
      //}


      //compute gradient for each point
      TPrecision maxL = 0;
      for(unsigned int j=0; j < Z.N(); j++){
        //compute gradient
        gradX(j, gx);
        
        //update if async updates
        if(async){
          Linalg<TPrecision>::ColumnAddScale(Z, j, -scaling, gx);
          //mark Zchanged in order to update f(Y) if needed
          Zchanged = true;
        }
        //store gradient if syncronous updates
        else{
          TPrecision l = Linalg<TPrecision>::Length(gx);
          if(l > maxL){
            maxL = l;
          }
          for(unsigned int k=0; k<Z.M(); k++){
            sync(k, j) = gx(k);
          }
        }
      }
      std::cout << std::endl;

      //Update if sync updates
      if(!async){
        //TPrecision avg = 0;
        TPrecision s = scaling * sX / maxL;
        //if(s < maxL){
        //  s = scaling * sX;
        //}
        //for(unsigned int i=0; i<fY.N(); i++){
        //  avg += KNNXD(1, i);
        //}
        //avg = avg/fY.N();
        //if(avg < maxL){
        // s = scaling * avg / maxL;
        //}

        std::cout << "scaling: " << s << std::endl;
        Linalg<TPrecision>::AddScale(Z, -s, sync, Z);
        Zchanged = true;
      }

      std::cout << std::endl;
      std::cout << "Iteration: " << i << std::endl;
      std::cout << "MSE: " <<  evaluate()/Y.N() << std::endl;



      //---Crossvalidation
      if(cvData.N() > 0 ){
        //project crossvalidation data
        parametrize(cvData, proj);
        reconstruct(proj, rcst);
        
        //compute mse
        Precision mse = 0;
        Precision var = 0;
        for(unsigned int j=0; j<cvData.N(); j++){
          se(j) = l2metric.distanceSquared(rcst, j, cvData, j);
          mse += se(j);
        }
        mse /= cvData.N();

        //Variance sanity check
        for(unsigned int j=0; j<cvData.N(); j++){
          TPrecision tmp = se(j) -  mse;
          var += tmp*tmp;
        }
        var /= (cvData.N()-1);

        std::cout << "CV MSE: " <<  mse << std::endl;
        std::cout << "CV Var SE: " <<  var << std::endl;

        //Check if cv mse is still decreasing - abort if increasing
        if(mseCVprev < mse){// || varCVprev < var){
          break;
        }
        varCVprev = var;
        mseCVprev = mse;
      }
      std::cout << std::endl << std::endl;
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
   void f( DenseVector<TPrecision> &y, DenseVector<TPrecision> &out ){
     Linalg<TPrecision>::Set(out, 0);


     //Compute gaussian kernel for y
     DenseVector<int> knns(std::max(knnY, knnSigma));
     DenseVector<TPrecision> knnDists(std::max(knnY, knnSigma));
     

     Geometry<TPrecision>::computeKNN(Y, y, knns, knnDists, l2metric);
     if(adaptive){
       kernelYtmp.setKernelParam(knnDists(knnSigma-1)); 
     }

     
     //do kernel regression 
     TPrecision sum = 0;
     for(unsigned int i=0; i<knnY; i++){
       wTmp(i) = kernelYtmp.f(y, Y, knns(i));
       sum += wTmp(i);
     }

     for(unsigned int i=0; i < knnY; i++){
       Linalg<TPrecision>::AddScale(out, wTmp(i)/sum, Z, knns(i), out); 
     }

     knns.deallocate();
     knnDists.deallocate();
   };




   //f(y_i) - coordinate mapping
   void f( int yi, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);
     TPrecision sum = 0;
     for(unsigned int i=0; i < knnY; i++){
       wTmp(i) = KY(KNNY(i, yi), yi) + lWeight * kernelL.f(L, i, L, KNNY(i, yi));
       sum += wTmp(i);
     }
     for(unsigned int i=0; i < knnY; i++){
        Linalg<TPrecision>::AddScale(out, wTmp(i)/sum, Z, KNNY(i, yi), out);
     }
   };
  

  //g(y_index) - reconstruction mapping
  void g( int index, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     //DenseVector<TPrecision> k(knnX);
     DenseVector<TPrecision> k(fY.N());
     TPrecision sum = 0;

     computefY();

     /* 
     for(unsigned int i=0; i < knnX; i++){  
       k(i) = kernelX.f(fY, KNNX(i, index), fY, index);
       sum += k(i);
     }

     for(unsigned int i=0; i < knnX; i++){
       Linalg<TPrecision>::AddScale(out, k(i)/sum, Y, KNNX(i, index), out); 
     } 
     */

     for(unsigned int i=0; i < fY.N(); i++){ 
       TPrecision d = sl2metric.distance(fY, i, fY, index);
       if(d>sX3s){
        k(i) = 0;
       }else{ 
        k(i) = kernelX.f(d);
        sum += k(i);
       }
     }

     for(unsigned int i=0; i < fY.N(); i++){
       if(k(i) != 0){
        Linalg<TPrecision>::AddScale(out, k(i)/sum, Y, i, out); 
       }
     } 

     k.deallocate();
   };


   //g(x) - reconstruction mapping
   void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     computefY();

     //DenseVector<int> knns(std::max(knnSigma, knnX));
     //DenseVector<TPrecision> knnDists(std::max(knnSigma, knnX));
     //Geometry<TPrecision>::computeKNN(fY, x, knns, knnDists, l2metric);

     //DenseVector<TPrecision> k(knnX);
     DenseVector<TPrecision> k(fY.N());
     TPrecision sum = 0;

     /*
     for(unsigned int i=0; i < knnX; i++){
       k(i) = kernelX.f(x, fY, knns(i));
       sum += k(i);
     }

     for(unsigned int i=0; i<knnX; i++){
        Linalg<TPrecision>::AddScale(out, k(i)/sum, Y, knns(i), out); 
     } 
     */

     for(unsigned int i=0; i < fY.N(); i++){ 
       TPrecision d = sl2metric.distance(fY, i, x); 
       //if(d>sX3s){
       // k(i) = 0;
       //}else{ 
        k(i) = kernelX.f(d);
        sum += k(i);
       //}
     }

     for(unsigned int i=0; i < fY.N(); i++){
       if(k(i)!=0){
         Linalg<TPrecision>::AddScale(out, k(i)/sum, Y, i, out); 
       }
     } 

     //knns.deallocate();
     //knnDists.deallocate();
     k.deallocate();
   };

   

   //Compute gradient of f, i.e. z_index 
   void gradX( int r, DenseVector<TPrecision> &gx){
     
     Linalg<TPrecision>::Zero(gx);
     
     //g(f(y_r))
     DenseVector<TPrecision> gfy(Y.M());

     //kernel values & derivatives for K(f(y_r) - f(y_j));
    // DenseVector<TPrecision> kx(knnX);
    // DenseMatrix<TPrecision> kxd(Z.M(), knnX);
     DenseVector<TPrecision> kx(fY.N());
     DenseMatrix<TPrecision> kxd(Z.M(), fY.N());
     
     //sum of kernel values
     DenseVector<TPrecision> sumkxd(Z.M());
     
     //gradient for each component of \hat{x}_r
     DenseMatrix<TPrecision> gradient(Y.M(), Z.M());

     //Temp vars.
     DenseVector<TPrecision> gxtmp(Z.M());
     DenseVector<TPrecision> diff(Y.M());
     TPrecision tmp;
     
     //update fY if necessary
     computefY();

     //Compute gradient
    for(unsigned int i=1; i < knnY; i++){


      int yi = KNNY(i, r);

      //x-kernel values & derivatves at f(y)
      TPrecision sumkx = 0;
      sumkxd.zero();

      //for(unsigned int j=0; j < knnX; j++){
      for(unsigned int j=0; j < fY.N(); j++){

        int yj = j;// KNNX(j, yi); 
        
        //derivative and kernel value
        if(yi == yj){
          kx(j) = 0;
          continue;
        }
        TPrecision d = sl2metric.distance(fY, yi, fY, yj);
        if(d>sX3s){
          kx(j) = 0;
          continue;
        }
        kx(j) = kernelX.gradf(fY, yi, fY, yj, gxtmp);

        //multipily kernel derivative by d[f(y_i)-f(y_j) / d z_r 
        //and store result in matrix kxd
        sumkx += kx(j);
        
        TPrecision df = KYN( yi, r ) - KYN( yj, r );
        for(unsigned int k=0; k < Z.M(); k++){
          kxd(k, j) = gxtmp(k) * df;
          sumkxd(k) += kxd(k, j);
        }
      }

      TPrecision sumkx2 = sumkx*sumkx;

      //g(f(y_i)) 
      Linalg<TPrecision>::Zero(gfy);
      //for(unsigned int j=0; j < knnX; j++){
      for(unsigned int j=0; j < fY.N(); j++){
        int yj = j;//KNNY(j, yi);
        if(kx(j)!=0){ 
          Linalg<TPrecision>::AddScale(gfy, kx(j)/sumkx, Y, yj, gfy); 
        }
      }

     
      //Gradient matrix of (g \cric f)(f(y)) for each component of z_r 
      Linalg<TPrecision>::Zero(gradient);
      //for(unsigned int j = 0; j < knnX; j++){
      for(unsigned int j = 0; j < fY.N(); j++){
        int yj = j;//KNNY(j, yi); 
        if(kx(j) == 0) continue;
        for(unsigned int n=0; n<Z.M(); n++){
          tmp =  ( kxd(n, j) * sumkx - kx(j) * sumkxd(n) ) / sumkx2;
          for(unsigned int m=0; m < Y.M() ; m++){
            gradient(m, n) +=  tmp * Y(m, yj);
          }
        }
      }
      

      //d E / d \hat{x}_r
      Linalg<TPrecision>::Subtract(gfy, Y, yi, diff);
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
   DenseMatrix<TPrecision> getZ(){
    return Z;
   };


   void chnageZ(){
     Zchanged = true;
   };






   //coordinate mapping fo Ypoints
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

  


   DenseMatrix<TPrecision> &parametrize(){
     computefY();
     return fY;
   };
  


   DenseMatrix<TPrecision> reconstruct(DenseMatrix<TPrecision> &Xpoints){
     DenseMatrix<TPrecision> proj(Y.M(), Xpoints.N());
     reconstruct(Xpoints, proj);     
     return proj;
   };



  

   void reconstruct(DenseMatrix<TPrecision> &Xpoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> tmp(Z.M()); 
     DenseVector<TPrecision> yp(Y.M()); 
     for(unsigned int i=0; i < Xpoints.N(); i++){
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

   GaussianKernel<Precision> getKernelX(){
     return kernelX;
   };


/*
   void change(DenseMatrix<TPrecision> Ynew, DenseMatrix<TPrecision> Znew,
       TPrecision alpha = -1){
     cleanup();
     Y = Ynew;
     Z = Znew;
     Zchanged = true;
     init();
     if(alpha > 0){
      computeKernelX(alpha);   
     }  
   };
   */


private:


  //void computeKNNX(){
    //std::cout << "updating knn X" << std::endl;
    //Geometry<TPrecision>::computeKNN(fY, KNNX, KNNXD, l2metric);
  //};


  void computefY(){
    if(Zchanged){
      std::cout << "updating f(Y)" << std::endl;
      Zchanged = false;
      DenseVector<TPrecision> tmp(Z.M());
      for(unsigned int i=0; i<Y.N(); i++){
        f(i, tmp);
        Linalg<TPrecision>::SetColumn(fY, i, tmp);
      }
      tmp.deallocate();
    }
  };
 


  void computeKernelX(TPrecision alpha){
    TPrecision sigma = 0;
    DenseMatrix<int> knns(knnSigma, fY.N());
    DenseMatrix<TPrecision> knnDists(knnSigma, fY.N());
    Geometry<TPrecision>::computeKNN(fY, knns, knnDists, l2metric);
    sX = 0;
    for(unsigned int i=0; i < fY.N(); i++){
      sigma += knnDists(knnSigma-1, i);
      sX += knnDists(1, i);

    }
    sigma *= alpha/fY.N();
    
    sX /= fY.N();
    sX3s = sigma*sigma*9;

    std::cout << "sigmaX: " << sigma << std::endl;
    std::cout << "scale: " << sX << std::endl;

    kernelX.setKernelParam(sigma);
    knns.deallocate();
    knnDists.deallocate();
  };


  

  void computeKY(){
    unsigned int N = Y.N();
    KY = DenseMatrix<TPrecision>(N, N);
    sumKY = DenseVector<TPrecision>(N);
    Linalg<TPrecision>::Set(sumKY, 0);
    KYN = DenseMatrix<TPrecision>(N, N);


    if(adaptive){
      for(unsigned int i=0; i<N; i++){
        kernelY[i] = GaussianKernel<TPrecision>(KNNYD(knnSigma-1, i), fY.M());
      }
    }
    else{
      Precision sigma = 0;
      for(unsigned int i=0; i<N; i++){
        sigma += KNNYD(knnSigma-1, i); 
      }
      sigma /= N;
      for(unsigned int i=0; i<N; i++){
        kernelY[i] = GaussianKernel<TPrecision>(sigma, fY.M());
      }
      kernelYtmp = GaussianKernel<TPrecision>(sigma, fY.M());
    }


    std::cout << "Compute KY" << std::endl;
    for(unsigned int i=0; i < N; i++){
      for(unsigned int j=0; j < N; j++){
        KY(j, i) = kernelY[i].f(Y, j, Y, i); 
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
