#ifndef LEVELSETKMM_H
#define LEVELSETKMM_H


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
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "FirstOrderLevelSetMetric.h"
#include "LevelSetKernelRegression.h"

#include <stdlib.h>
#include <limits>
#include <math.h>


template <typename TPrecision, typename TImage>
class LevelSetKMM{
  
  private:

    typedef typename TImage::Pointer TImagePointer;

    typedef typename itk::GradientMagnitudeImageFilter<TImage, TImage>
      GradientMagnitudeFilter;
    typedef typename GradientMagnitudeFilter::Pointer
      GradientMagnitudeFilterPointer;

    typedef typename itk::ImageRegionConstIterator<TImage> ImageRegionConstIterator;
    typedef typename itk::ImageRegionIterator<TImage> ImageRegionIterator;

    TPrecision sX;

    bool Zchanged;
    bool verbose;


  protected:    
    DenseVector<TImagePointer> Y;
    DenseMatrix<TPrecision> Z;
    DenseMatrix<TPrecision> fY;

    unsigned int knnSigma;
    unsigned int knnY;
    unsigned int knnX;
    
    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;
    FirstOrderLevelSetMetric<TPrecision, TImage> lsmetric;
    LevelSetKernelRegression<TPrecision, TImage> lsr;

    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;


    DenseMatrix<int> KNNY;
    DenseMatrix<TPrecision> KNNYD;


    DenseMatrix<int> KNNX;
    DenseMatrix<TPrecision> KNNXD;
    
    GaussianKernel<TPrecision> kernelX;
    GaussianKernel<TPrecision> kernelY;
  
  public:
  
   virtual void cleanup(){      
    KNNX.deallocate();
    KNNXD.deallocate();    
    KNNY.deallocate();
    KNNYD.deallocate();
    KY.deallocate();
    sumKY.deallocate();
    KYN.deallocate();
    Z.deallocate();
    fY.deallocate();
   };

   //Create KernelMap 
   LevelSetKMM(DenseVector<TImagePointer> Ydata, DenseMatrix<TPrecision> Zinit, 
       TPrecision alpha, unsigned int nnSigma, unsigned int nnY, unsigned int
       nnX) :
       Y(Ydata), Z(Zinit), knnSigma(nnSigma), knnY(nnY), knnX(nnX){
     
     verbose = true;//false;
     init();
     computeKernelX(alpha);
   };



   LevelSetKMM(DenseVector<TImagePointer> Ydata, DenseMatrix<TPrecision> Zopt, 
       unsigned int nnSigma, unsigned int nnY, unsigned int nnX,double sigmaX): Y(Ydata),
       Z(Zopt), knnSigma(nnSigma), knnY(nnY), knnX(nnX){

     verbose = true;//false;
     init();
     kernelX.setKernelParam(sigmaX);   

     if(verbose){
     std::cout << "knnSigma: " << knnSigma<<std::endl;
     std::cout << "knnX: " << nnX <<std::endl;
     std::cout << "knnY: " << nnY <<std::endl;
     std::cout << "sigmaX: " << kernelX.getKernelParam() <<std::endl;
     std::cout << "sigmY: " << kernelY.getKernelParam() <<std::endl;
     std::cout << "adaptive: " << false <<std::endl;
     }

   }; 


   
 



   //evalue objective function, squared error
   virtual TPrecision mse(){
     computefY();
     TPrecision e = 0;
     
     //Temp vars.
     TImagePointer gfy = ImageIO<TImage>::copyImage(Y(0));

     for(unsigned int i=0; i < Y.N(); i++){
       g(i, gfy);
       e += lsmetric.distance(Y(i), gfy);
     }

     return e/Y.N();
   };


   virtual TPrecision mse(int index){
     computefY();
     TPrecision e = 0;
     
     
     //Temp vars.
     TImagePointer gfy  = ImageIO<TImage>::copyImage(Y(0));

     for(unsigned int i=0; i < knnY; i++){
       int nn = KNNY(i, index);
       g(nn, gfy);
       e += lsmetric.distance(Y(nn), gfy); 
     }
          
     return e/knnY;
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
      startIteration(i);
      //compute gradient for each point
      TPrecision maxL = 0;
      for(unsigned int j=0; j < Z.N(); j++){
        //compute gradient
        //gradX(j, gx);
	      gradX(j, gx, sX/10.f);
        
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

      //sync updates
      TPrecision s;
      if(maxL == 0 )
	s = scaling;
      else{
        s = scaling * sX/maxL;
      }     
      if(verbose){
        std::cout << "scaling: " << s << std::endl;
      }
      
      //evaluate();
      
      //Linalg<TPrecision>::AddScale(Z, -s, sync, Z);

      
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
      endIteration(i);



      std::cout << std::endl;
      TPrecision obj = mse(); 
      //TPrecision ortho = orthogonality(); 
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
 

 


  //f(x_index) - reconstruction mapping
  void f(unsigned int index, Vector<TPrecision> &out){
    Linalg<TPrecision>::Zero(out);
    TPrecision sumw = 0;
    for(unsigned int i=0; i < Y.N(); i++){
      TPrecision w = KY(i, index);
      Linalg<TPrecision>::AddScale(out, w, Z, i, out);
      sumw += w;
    }     
    Linalg<TPrecision>::Scale(out, 1.f/sumw, out);
  };




   //f(x) - reconstruction mapping
   void f( ImagePointer &y, Vector<TPrecision> &out){
     //DenseVector<int> knn(knnY);
     //DenseVector<TPrecision> knnDist(knnY);
       
     TPrecision dists[Y.N()];
     for(unsigned int i = 0; i < Y.N(); i++){
       dists[i] = lsmetric.distance(y, Y(i)); 
        
       //MinHeap<TPrecision> minHeap(dists, Y.N());
       //for(unsigned int j=0; j < knn.N(); j++){
       //  knn(j) = minHeap.getRootIndex();
       //  knnDist(j) = minHeap.extractRoot();
       //}
     }


     Linalg<TPrecision>::Zero(out);
     TPrecision sumw = 0;
     for(unsigned int i=0; i < Y.N(); i++){
       TPrecision w = kernelY.f(dists[i]*dists[i]);
       Linalg<TPrecision>::AddScale(out, w, Z, i, out);
       sumw += w;
     }     
     Linalg<TPrecision>::Scale(out, 1.f/sumw, out);

     //knn.deallocate();
     //knnDist.deallocate();
   };






  
  //------g linear 
  //g(x_index) - reconstruction mapping
  void g(unsigned int index, ImagePointer out){
     computefY();
     DenseVector<TPrecision> x(fY.M());
     Linalg<TPrecision>::ExtractColumn(fY, index, x);
     lsr.evaluate(x, out, index);
     x.deallocate();
   };  
   


   //g(x) - reconstruction mapping
   void g( Vector<TPrecision> &x, ImagePointer out){
     computefY();
     lsr.evaluate(x, out);
   };   
  



  


   //numerical gradient computation
   virtual void gradX(int r, DenseVector<TPrecision> &gx, TPrecision epsilon){
    //TPrecision e = evaluate(r);
    TPrecision eg = 0;
    TPrecision e = mse(r);
    DenseVector<TPrecision> fy(Z.M());
    for(unsigned int i=0; i<gx.N(); i++){
      Z(i, r) += epsilon;
      //update nearest neighbors
      for(unsigned int k=0; k<knnX; k++){
	      int nn = KNNX(k, r);
	      f(nn, fy);
        Linalg<TPrecision>::SetColumn(fY, nn, fy);
      }
      eg = mse(r);
      gx(i) = ( eg - e ) / epsilon;
      Z(i, r) -= epsilon;
    }
    
    //update nearest neighbors
    for(unsigned int k=0; k<knnX; k++){
      int nn = KNNX(k, r);
      f(nn, fy);
      Linalg<TPrecision>::SetColumn(fY, nn, fy);
    }

    fy.deallocate();
   };




   int getKnnY(){
     return knnY;
   }
   
   int getKnnX(){
     return knnX;
   }

   //get original Y's
   DenseMatrix<TPrecision> getY(){
     return Y;
   };
   


   //get Z (parameters for f
   DenseMatrix<TPrecision> getZ(){
     return Z;
   };


   //coordinate mapping fo Ypoints
   DenseMatrix<TPrecision> parametrize(DenseVector<TImagePointer> &Ypoints){

     DenseMatrix<TPrecision> proj(Z.M(), Ypoints.N());
     parametrize(Ypoints, proj);

     return proj;
   };


   //
   void parametrize(DenseVector<TImagePointer> &Ypoints, DenseMatrix<TPrecision> &proj){

     DenseVector<TPrecision> xp(Z.M()); 

     for(unsigned int i=0; i < Ypoints.N(); i++){
       f(Ypoints(i), xp);
       Linalg<TPrecision>::SetColumn(proj, i, xp);
     }
     xp.deallocate();
   };

  


   DenseMatrix<TPrecision> &parametrize(){
     computefY();
     return fY;
   };
  


   DenseVector<TImagePointer> reconstruct(DenseMatrix<TPrecision> &Xpoints){
     DenseVector<TImagePointer> proj(Xpoints.N());
     reconstruct(Xpoints, proj);     
     return proj;
   };



  

   void reconstruct(DenseMatrix<TPrecision> &Xpoints, DenseVector<TImagePointer> &proj){

     DenseVector<TPrecision> tmp(Z.M()); 
     for(unsigned int i=0; i < Xpoints.N(); i++){
       Linalg<TPrecision>::ExtractColumn(Xpoints, i, tmp);
       TImagePointer yp = ImageIO<TImage>::copyImage(Y(0));
       g(tmp, yp);
       proj(i) = yp;
     }
     tmp.deallocate();
   };   
   


   TPrecision getSigmaX(){
     return kernelX.getKernelParam();
   };

   GaussianKernel<Precision> getKernelX(){
     return kernelX;
   };


   void setStep(TPrecision s){
     lsr.setStep(s);
   };


   void setNIter(unsigned int n){
     lsr.setNIter(n);
   };


   void setStopping(TPrecision stop){
     lsr.setStopping(stop);
   };


   void setEpsilon(TPrecision e){
     lsmetric.setEpsilon(e);
     lsr.setEpsilon(e);
   };    

protected:
   virtual void startIteration(int iter){};
   virtual void endIteration(int iter){};


private:


   void init(){

     if(verbose){
      std::cout << "Initalizing" << std::endl;
     }
    
     kernelY = GaussianKernel<TPrecision>( Z.M());
     kernelX = GaussianKernel<TPrecision>( Z.M());
     
     if(knnY >= Y.N()){
      knnY = Y.N()-1;
     }
     if(knnX >= Y.N()){
      knnX = Y.N()-1;
     }
     if(knnSigma >= Y.N()){
       knnSigma = Y.N()-1;
     }

     if(verbose){
      std::cout << "Computing knn for Y" << std::endl;
     }

     
     KNNX =  DenseMatrix<int>(knnX+1, Y.N());
     KNNXD = DenseMatrix<TPrecision>(knnX+1, Y.N());
      
     KNNY =  DenseMatrix<int>(knnY+1, Y.N());
     KNNYD = DenseMatrix<TPrecision>(knnY+1, Y.N());

     if(verbose){
      std::cout << "Computing kernel values for Y" << std::endl;
     }

     computeKY();

     fY = Linalg<TPrecision>::Copy(Z);
     lsr = LevelSetKernelRegression<TPrecision, TImage>(Y, fY, kernelX);
     lsr.setLevelSetMetric(lsmetric);
     Zchanged = true;
     computefY(); 

     if(verbose) {   
      std::cout << "Initalizing done" << std::endl;
     }

   };     

  void computeNNX(){
    Geometry<TPrecision>::computeKNN(fY, KNNX, KNNXD, sl2metric);
  };


  void computefY(){
    if(Zchanged){
      Zchanged = false;
      DenseVector<TPrecision> tmp(Z.M());
      for(unsigned int i=0; i<Y.N(); i++){
        f(i, tmp);
        Linalg<TPrecision>::SetColumn(fY, i, tmp);
      }
      tmp.deallocate();
      computeNNX();
    }
  };
 


  void computeKernelX(TPrecision alpha){
    TPrecision sigma = 0;
    computefY();
    DenseMatrix<int> knns(knnSigma, fY.N());
    DenseMatrix<TPrecision> knnDists(knnSigma, fY.N());
    Geometry<TPrecision>::computeKNN(fY, knns, knnDists, sl2metric);
    sX = 0;
    for(unsigned int i=0; i < fY.N(); i++){
      sigma += sqrt( knnDists(knnSigma-1, i) );
      sX += sqrt(knnDists(1, i));

    }
    sigma *= alpha/fY.N();
    
    sX /= fY.N();

    if(verbose){
      std::cout << "sigmaX: " << sigma << std::endl;
      std::cout << "scale: " << sX << std::endl;
    }

    kernelX.setKernelParam(sigma);
    knns.deallocate();
    knnDists.deallocate();

    lsr.setKernelSigma(sigma);
  };


  

  void computeKY(){
    unsigned int N = Y.N();
    KY = DenseMatrix<TPrecision>(N, N);
    sumKY = DenseVector<TPrecision>(N);
    Linalg<TPrecision>::Set(sumKY, 0);
    KYN = DenseMatrix<TPrecision>(N, N);

    DenseMatrix<TPrecision> distances(Y.N(), Y.N());
    for(unsigned int i=0; i < N; i++){
      for(unsigned int j=i; j < N; j++){
        distances(i, j) = lsmetric.distance(Y(i), Y(j));
        distances(j, i) = distances(i, j);
      }
    }
        
    
    TPrecision dists[Y.N()];
    for(unsigned int i = 0; i < Y.N(); i++){
      for(unsigned int j=0; j<Y.N(); j++){
        dists[j] = distances(j, i); 
      }
        
      MinHeap<TPrecision> minHeap(dists, Y.N());
      for(unsigned int j=0; j < KNNY.M(); j++){
        KNNY(j, i) = minHeap.getRootIndex();
        KNNYD(j, i) = minHeap.extractRoot();
      }
    }
    
    


    
    Precision sigma = 0;
    for(unsigned int i=0; i<N; i++){
      sigma += sqrt( KNNYD(knnSigma-1, i) ); 
    }
    sigma /= (3*N);
    if(verbose){
      std::cout << "sigmaY: " << sigma << std::endl;
    }
    kernelY.setKernelParam(sigma);




    if(verbose){
      std::cout << "Compute KY" << std::endl;
    }
    for(unsigned int i=0; i < N; i++){
      for(unsigned int j=0; j < N; j++){
        KY(j, i) = kernelY.f(distances(i, j) * distances(i, j)); 
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

