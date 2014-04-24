#ifndef WEIGHTEDKMM_H
#define WEIGHTEDKMM_H

#include "Geometry.h"
#include "Matrix.h"
#include "EuclideanMetric.h"
#include "SquaredEuclideanMetric.h"
#include "WeightedKernelDensity.h"
#include "Linalg.h"
#include "GaussianKernel.h"

#include <stdlib.h>
#include <limits>

template <typename TPrecision>
class WeightedKMM{

  private:
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> Z;
    DenseMatrix<TPrecision> fY;

    DenseVector<TPrecision> w;

    unsigned int knnSigma;
    unsigned int knnY;
    unsigned int knnX;

    unsigned int knnXUpdate;


    DenseMatrix<int> KNNY;
    DenseMatrix<TPrecision> KNNYD;

    DenseMatrix<int> KNNX;
    DenseMatrix<TPrecision> KNNXD;

    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;

    GaussianKernel<TPrecision> kernelX;
    GaussianKernel<TPrecision> *kernelY;
    GaussianKernel<TPrecision> kernelYtmp;
    
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;

    bool Zchanged;

    bool adaptive;



  public:
  
   void cleanup(){
    KNNY.deallocate();
    KNNYD.deallocate();
    KNNX.deallocate();
    KNNXD.deallocate();
    KY.deallocate();
    sumKY.deallocate();
    KYN.deallocate();
    Y.deallocate();
    Z.deallocate();
    fY.deallocate();
    w.deallocate();
    delete[] kernelY; 
   };

   //Create KernelMap 
   WeightedKMM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zinit,
       DenseVector<TPrecision> weights, TPrecision alpha, unsigned int nnSigma,
       unsigned int nnY, unsigned int nnX, bool adapt = false) :
       Y(Ydata), Z(Zinit), w(weights), knnSigma(nnSigma), knnY(nnY), knnX(nnX), adaptive(adapt){
     
     init();
     computeKernelX(alpha);
   };



   WeightedKMM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zopt,
       DenseVector<TPrecision> weights, 
       unsigned int nnSigma, unsigned int nnY, unsigned int nnX, 
       double sigmaX, bool adapt = false): Y(Ydata),
       Z(Zopt), w(weights), knnSigma(nnSigma),knnY(nnY), knnX(nnX), adaptive(adapt){
     init();
     kernelX.setKernelParam(sigmaX);
   }; 


   void init(){
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

     knnXUpdate = 6;

     knn = std::max(knnSigma, knnX);
     KNNX = DenseMatrix<int>(knn, Y.N());
     KNNXD = DenseMatrix<TPrecision>(knn, Y.N());

     std::cout << "Computing kernel values for Y" << std::endl;
     computeKY();

     fY = DenseMatrix<TPrecision>(Z.M(), Z.N()); 
     Zchanged = true;
     computefY();

     computeKNNX();
     std::cout << "Initalizing done" << std::endl;
   };     
   

   //evalue objective function, squared error
   TPrecision evaluate(){
     TPrecision e = 0;
     TPrecision sum = 0;

     computefY();
     DenseVector<TPrecision> gfy(Y.M());
     for(unsigned int i=0; i < Y.N(); i++){
        g(i, gfy);
        sum += w(i);
        e += w(i) * l2metric.distanceSquared(Y, i, gfy); 
     }
     e /= sum;

     gfy.deallocate();

     return e;
   };
   

   //Gradient descent for all points 
   void gradDescent(unsigned int nIterations, TPrecision scaling, 
       DenseMatrix<TPrecision> &cvData, DenseVector<TPrecision> &wcv, bool async = false){
     


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
      if( (i+1) % knnXUpdate == 0){
        computeKNNX();
      }

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

      //Update if sync updates
      if(!async){
        TPrecision avg = 0;
        for(unsigned int i=0; i<fY.N(); i++){
          avg += KNNXD(1, i);
        }
        scaling = (avg/fY.N())/maxL;
        std::cout << "scaling: " << scaling << std::endl;
        Linalg<TPrecision>::AddScale(Z, -scaling, sync, Z);
        Zchanged = true;
      }

      std::cout << std::endl;
      std::cout << "Iteration: " << i << std::endl;
      std::cout << "MSE: " <<  evaluate() << std::endl;



      //---Crossvalidation
      if(cvData.N() > 0 ){
        //project crossvalidation data
        parametrize(cvData, proj);
        reconstruct(proj, rcst);
        
        //compute mse
        Precision mse = 0;
        Precision var = 0;
        Precision sum = 0;
        for(unsigned int j=0; j<cvData.N(); j++){
          se(j) = wcv(i) * l2metric.distanceSquared(rcst, j, cvData, j);
          mse += se(j);
          sum += wcv(i);
        }
        mse /= sum;

        //Variance sanity check
        for(unsigned int j=0; j<cvData.N(); j++){
          TPrecision tmp = se(j)/sum -  mse;
          var += tmp*tmp;
        }
        var /= (cvData.N()-1);

        std::cout << "CV MSE: " <<  mse << std::endl;
        std::cout << "CV Var SE: " <<  var << std::endl;

        //Check if cv mse is still decreasing - abort if increasing
        if(mseCVprev < mse){
          break;
        }

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
     DenseVector<TPrecision> k(knnY);
     TPrecision sum = 0;
     for(unsigned int i=0; i<knnY; i++){
       k(i) = w(knns(i)) * kernelYtmp.f(y, Y, knns(i)); 
       sum += k(i);
     }

     for(unsigned int i=0; i < knnY; i++){
       Linalg<TPrecision>::AddScale(out, k(i) / sum, Z, knns(i), out); 
     }

     knns.deallocate();
     knnDists.deallocate();
     k.deallocate();  
   };




   //f(y_i) - coordinate mapping
   void f( int yi, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);
     
     TPrecision sum = 0;
     for(unsigned int i=0; i < knnY; i++){
       sum += KY(KNNY(i, yi), yi);
     }

     for(unsigned int i=0; i < knnY; i++){
        Linalg<TPrecision>::AddScale(out, KY( KNNY(i, yi), yi ) / sum, Z, KNNY(i, yi), out);
     }
   };
  

  //g(y_index) - reconstruction mapping
  void g( int index, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     DenseVector<TPrecision> k(knnX);
     TPrecision sum = 0;

     computefY();
     for(unsigned int i=0; i < knnX; i++){  
       k(i) =  w(KNNX(i, index)) * kernelX.f(fY, KNNX(i, index), fY, index);
       sum += k(i);
     }

     for(unsigned int i=0; i < knnX; i++){
       Linalg<TPrecision>::AddScale(out, k(i) / sum, Y, KNNX(i, index), out); 
     } 

     k.deallocate();
   };


   //g(x) - reconstruction mapping
   void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
     Linalg<TPrecision>::Set(out, 0);

     computefY();

     DenseVector<int> knns(std::max(knnSigma, knnX));
     DenseVector<TPrecision> knnDists(std::max(knnSigma, knnX));
     Geometry<TPrecision>::computeKNN(fY, x, knns, knnDists, l2metric);

     DenseVector<TPrecision> k(knnX);
     TPrecision sum = 0;

     for(unsigned int i=0; i < knnX; i++){
       k(i) = w(knns(i)) * kernelX.f(x, fY, knns(i));
       sum += k(i);
     }

     for(unsigned int i=0; i<knnX; i++){
        Linalg<TPrecision>::AddScale(out, k(i) / sum, Y, knns(i), out); 
     } 

     knns.deallocate();
     knnDists.deallocate();
     k.deallocate();
   };

   

   //Compute gradient of f, i.e. z_index 
   void gradX( int index, DenseVector<TPrecision> &gx){
     
     Linalg<TPrecision>::Zero(gx);
     
     //g(f(y_index))
     DenseVector<TPrecision> gfy(Y.M());

     //kernel values & derivatives for K(f(y_index) - f(y_j));
     DenseVector<TPrecision> kx(knnX);
     DenseMatrix<TPrecision> kxd(Z.M(), knnX);

     //sum of kernel values
     DenseVector<TPrecision> sumkxd(Z.M());
     
     //gradient for each component of \hat{x}_index
     DenseMatrix<TPrecision> gradient(Y.M(), Z.M());

     //Temp vars.
     DenseVector<TPrecision> gxtmp(Z.M());
     DenseVector<TPrecision> diff(Y.M());
     TPrecision tmp;
     
     //update fY if necessary
     computefY();


     for(unsigned int i=0; i < knnY; i++){
      int yi = KNNY(i, index);

      //x-kernel values & derivatves at f(y)
      TPrecision sumkx = 0;
      sumkxd.zero();

      for(unsigned int j=0; j < knnX; j++){

        int yj = KNNX(j, yi); 
        
        //derivative and kernel value
        kx(j) = w(yj) * kernelX.gradf(fY, yi, fY, yj, gxtmp);

        //multipily kernel derivative by d[f(y_i)-f(y_j) / dx_index 
        //and store result in matrix kxd
        sumkx += kx(j);
        if(kx(j) == 0 || i == j){
          for(unsigned int k=0; k<Z.M(); k++){
            kxd(k, j) = 0;
          }
        }
        else{
          TPrecision df = KYN( yi, index ) - KYN( yj, index );
          for(unsigned int k=0; k < Z.M(); k++){
            kxd(k, j) = w(yj) * gxtmp(k) * df;
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
      for(unsigned int j=0; j < knnX; j++){
        int yj = KNNX(j, yi); 
        Linalg<TPrecision>::AddScale(gfy, kx(j) / sumkx, Y, yj, gfy); 
      }

     
      //Gradient matrix of (g \cric f)(f(y)) for each component of z_index 
      Linalg<TPrecision>::Zero(gradient);
      for(unsigned int j = 0; j < knnX; j++){
        int yj = KNNX(j, yi); 
        for(unsigned int n=0; n<Z.M(); n++){
          tmp =  ( kxd(n, j) * sumkx - kx(j) * sumkxd(n) ) / sumkx2;
          for(unsigned int m=0; m < Y.M() ; m++){
            gradient(m, n) +=  tmp * Y(m, yj);
          }
        }
      }
      

      //d E / d \hat{x}_index
      Linalg<TPrecision>::Subtract(gfy, Y, yi, diff);
      Linalg<TPrecision>::Scale(diff, 2 * w(yi), diff);
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


   //Compute the log probability of Yt_i belonging to this manifold, by computing a
   //local variance orthogonal to the manifold. The probability is the a
   //gaussian according to this variance and mean zero off the manifold
   //-Yt  data to test
   //-Ytp projection of Yt onto this manifold
   //-Xt manifold paprametrization of Yt
   //-Yp rpojection of the training data set onto this manifold
   //-p - output of pdf values for each point
   //-var - variances for each point
   void pdf(DenseMatrix<TPrecision> Yt, DenseMatrix<TPrecision> Ytp,
       DenseMatrix<TPrecision> Xt, DenseMatrix<TPrecision> Yp,
       DenseVector<TPrecision> &p, DenseVector<TPrecision> &var, bool
       useDensity){

     //update fY if necessary
     computefY();

     TPrecision cod = (TPrecision) (Y.M() - fY.M())/2.0;

     //compute trainig set squared distances
     DenseVector<TPrecision> sdist(Y.N());
     for(unsigned int i=0; i< Y.N(); i++){
       sdist(i) = sl2metric.distance(Y, i, Yp, i);
     }

     DenseVector<int> knns(std::max(knnSigma, knnX));
     DenseVector<TPrecision> knnDists(std::max(knnSigma, knnX));
     DenseVector<TPrecision> k(knnX);

     //compute variances and pdf values
     TPrecision c = - cod * log(2*M_PI);
     DenseVector<TPrecision> xt(Xt.M());

     for(unsigned int i=0; i < Xt.N(); i++){

       Linalg<TPrecision>::ExtractColumn(Xt, i, xt);
       Geometry<TPrecision>::computeKNN(fY, xt, knns, knnDists, l2metric);

       TPrecision sum = 0;
       TPrecision vartmp = 0;
       for(unsigned int j=0; j < knnX; j++){
         k(j) = w(knns(j)) *kernelX.f(Xt, i, fY, knns(j));
         sum += k(j);
         vartmp += sdist(knns(j)) * k(j); 
       } 
       var(i) = vartmp / sum;
       
       TPrecision d = sl2metric.distance(Yt, i, Ytp, i);
       p(i) = c - cod * log(var(i)) - d / ( 2 * var(i) ) ;
     }
     
     if(useDensity){
       TPrecision normalize = 0;
       for(unsigned int i=0; i< w.N(); i++){
         normalize += w(i);
       }
       normalize = log(normalize);
       WeightedKernelDensity<Precision> kd(fY, kernelX, w);
       for(unsigned int i=0; i<p.N(); i++){
         p(i) += log( kd.p(Xt, i, true)) - normalize;
       }
     }


     xt.deallocate(); 
     sdist.deallocate(); 
     knns.deallocate();
     knnDists.deallocate();
     k.deallocate();


   };


   TPrecision getSigmaX(){
     return kernelX.getKernelParam();
   };



   void setWeights(DenseVector<TPrecision> weights){
     w.deallocate();
     w = weights;
     computeKY();
   };

   void updateZ(){
      Z.deallocate();
      Z = Linalg<TPrecision>::Copy(fY);      
      Zchanged = true;
      computefY();
      computeKNNX();
      computeKernelX(1);
   };


private:


  void computeKNNX(){
    std::cout << "updating knn X" << std::endl;
    Geometry<TPrecision>::computeKNN(fY, KNNX, KNNXD, l2metric);
  };


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
    TPrecision sum = 0;
    for(unsigned int i=0; i < fY.N(); i++){
      sigma += w(i) * KNNXD(knnSigma-1, i);
      sum += w(i);
    }
    sigma/=sum;
    kernelX.setKernelParam(alpha*sigma);
  };




  void computeKY(){
    unsigned int N = Y.N();
    KY = DenseMatrix<TPrecision>(N, N);
    sumKY = DenseVector<TPrecision>(N);
    Linalg<TPrecision>::Set(sumKY, 0);
    KYN = DenseMatrix<TPrecision>(N, N);


    if(adaptive){
      for(unsigned int i=0; i<N; i++){
        kernelY[i].setKernelParam(KNNYD(knnSigma-1, i)); 
      }
    }
    else{
      TPrecision sum = 0;
      TPrecision sigma = 0;
      for(unsigned int i=0; i<N; i++){
        sigma += w(i) * KNNYD(knnSigma-1, i); 
        sum +=  w(i);
      }
      sigma /= sum;
      for(unsigned int i=0; i<N; i++){
        kernelY[i].setKernelParam(sigma); 
      }
      kernelYtmp.setKernelParam(sigma);
    }


    std::cout << "Compute KY" << std::endl;
    for(unsigned int i=0; i < N; i++){
      sumKY(i) = 0;
      for(unsigned int j=0; j < N; j++){
        KY(j, i) = w(j) * kernelY[i].f(Y, j, Y, i); 
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
