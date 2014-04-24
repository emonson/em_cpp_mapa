#ifndef CEM_H
#define CEM_H


#include "Geometry.h"
#include "Matrix.h"
#include "EuclideanMetric.h"
#include "SquaredEuclideanMetric.h"
#include "KernelDensity.h"
#include "Linalg.h"
#include "GaussianKernel.h"
#include "MahalanobisKernel.h"

#include <list>
#include <iterator>
#include <stdlib.h>
#include <limits>
#include <math.h>

//Minimize MSE
#define CEM_MSE 0
//Minimize <g(f(y)) - y , g'(f(y))>
#define CEM_ORTHO 1
//Minimize <g(f(y)) - y , g'(f(y))> with g'(f(y)) normalized
#define CEM_ORTHO_NORMALIZE 2 
//Minimize <g(f(y)) - y , g'(f(y))> with g'(f(y)) and g(f(y)) - y  normalized
#define CEM_ORTHO_NORMALIZE2 3 

//Conditional Expectation Manifolds
template <typename TPrecision>
class CEM{

  private:    
    DenseMatrix<TPrecision> Y;
    DenseMatrix<TPrecision> Z;
    DenseMatrix<TPrecision> fY;

    unsigned int knnY;
    unsigned int knnX;

    EuclideanMetric<TPrecision> l2metric;
    SquaredEuclideanMetric<TPrecision> sl2metric;


    DenseMatrix<int> KNNY;
    DenseMatrix<TPrecision> KNNYD;
    DenseMatrix<TPrecision> KY;
    DenseVector<TPrecision> sumKY;
    DenseMatrix<TPrecision> KYN;


    DenseMatrix<int> KNNX;
    DenseMatrix<TPrecision> KNNXD;
    DenseMatrix<TPrecision> KX;
    DenseVector<TPrecision> sumKX;
    DenseMatrix<TPrecision> KXN;



    GaussianKernel<TPrecision> kernelX;
    GaussianKernel<TPrecision> kY;
    MahalanobisKernel<TPrecision> *kernelY;

    TPrecision sX;



  public:

    
    virtual void cleanup(){      
      partialCleanup();
      Y.deallocate();
      Z.deallocate();

    };

    void partialCleanup(){  
      KNNX.deallocate();
      KNNXD.deallocate();	   
      KX.deallocate();
      sumKX.deallocate();
      KXN.deallocate();
      KNNY.deallocate();
      KNNYD.deallocate();
      KY.deallocate();
      sumKY.deallocate();
      KYN.deallocate();
      fY.deallocate();
      delete[] kernelY;
    };


    //Create Condtional Expectation Manifold 
    CEM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zinit, 
        TPrecision alpha, unsigned int nnY, unsigned int nnX) :
      Y(Ydata), Z(Zinit), knnY(nnY), knnX(nnX){

        init();
        initKY();
        update();
        computeKernelX(alpha);
        updateKNNX();
        updateKY();
        update();
      };



    CEM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Zopt, 
        unsigned int nnY, unsigned int nnX, double sigmaX,
        MahalanobisKernel<TPrecision> *kY): Y(Ydata), Z(Zopt), knnY(nnY),
    knnX(nnX), kernelY(kY) {

      init();
      //initKY();
      kernelX.setKernelParam(sigmaX);
      sX = kernelX.getKernelParam(); 
      //updateKNNX(); 
      updateKY(false);
      update();

    }; 





    //evalue objective function, squared error
    TPrecision mse(){
      TPrecision e = 0;
      DenseVector<TPrecision> gfy(Y.M());
      for(unsigned int i=0; i < Y.N(); i++){
        g(i, gfy);
        e += sl2metric.distance(Y, i, gfy);
      }
      gfy.deallocate();
      return e/Y.N();
    }


    //evalue objective function, squared error
    TPrecision mse(TPrecision &o, int type){
      o=0;
      TPrecision e = 0;

      //Jacobian of g(x)
      DenseMatrix<TPrecision> J(Y.M(), Z.M());

      //Temp vars.
      DenseVector<TPrecision> gfy(Y.M());
      DenseVector<TPrecision> diff(Y.M());
      DenseVector<TPrecision> pDot(Z.M());

      for(unsigned int i=0; i < Y.N(); i++){
        g(i, gfy, J);
        //e += sl2metric.distance(Y, i, gfy);

        Linalg<TPrecision>::Subtract(gfy, Y, i, diff);  
        e += Linalg<TPrecision>::SquaredLength(diff);

        if(type == CEM_ORTHO_NORMALIZE){
          Linalg<TPrecision>::QR_inplace(J);
        }
        if(type == CEM_ORTHO_NORMALIZE2){
          Linalg<TPrecision>::Normalize(diff);
        }
        Linalg<TPrecision>::Multiply(J, diff, pDot, true);

        for(unsigned int n=0; n< pDot.N(); n++){
          //o += acos(sqrt(pDot(n)*pDot(n)));
          o += pDot(n) * pDot(n);
        }  
      }
      //o = o/(Z.M()*Z.N())/ M_PI * 180;
      o = o/Z.N();

      pDot.deallocate();
      gfy.deallocate();
      diff.deallocate();
      J.deallocate();


      return e/Y.N();
    };







    //Gradient descent 
    void gradDescent(unsigned int nIterations, TPrecision scaling, int verbose=1, int type = CEM_ORTHO){

      TPrecision orthoPrev =0;
      TPrecision ortho;

      TPrecision objPrev = mse(orthoPrev, type);

      if(verbose > 0){
        std::cout << "Mse start: " << objPrev << std::endl;
        std::cout << "Ortho start: " << orthoPrev << std::endl;
      }

      if(verbose > 1){
        std::cout << "Kernel X sigma: " << kernelX.getKernelParam() << std::endl; 
      }


      //---Storage for syncronous updates 
      DenseMatrix<TPrecision> sync(Z.M(), Z.N());

      //---Do nIterations of gradient descent     
      DenseMatrix<TPrecision> Ztmp(Z.M(), Z.N());
      DenseMatrix<TPrecision> Zswap;

      //gradient direction
      DenseVector<TPrecision> gx(Z.M());     
      if(verbose > 0){
        std::cout << "Start Gradient Descent" << std::endl << std::endl;
      }

      for(unsigned int i=0; i<nIterations; i++){
        updateKY();
        update();
        //compute gradient for each point
        TPrecision maxL = 0;
        for(unsigned int j=0; j < Z.N(); j++){
          //compute gradient
          //gradX(j, gx);
          numGradX(j, gx, sX*0.1, type);
          

          //Asynchronous updates
          //Linalg<TPrecision>::ColumnAddScale(Z, j, -scaling*sX, gx);

          
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






        //sync updates
        TPrecision s;
        if(maxL == 0 )
          s = scaling;
        else{
          s = scaling * sX/maxL;
        }     
        if(verbose > 1){
          std::cout << "scaling: " << s << std::endl;
        }


        //Approximate line search with quadratic fit
        DenseMatrix<TPrecision> A(3, 3);
        DenseMatrix<TPrecision> b(3, 1);
        Linalg<TPrecision>::Zero(A);
        double orthoTmp = 0;
        double mseTmp = mse(orthoTmp, type);
        if(type == CEM_MSE){
          b(0, 0) = mseTmp;
        }
        else{
          b(0, 0) = orthoTmp;
        }

        Linalg<TPrecision>::AddScale(Z, -1*s, sync, Ztmp);
        Zswap = Z;
        Z = Ztmp;
        Ztmp = Zswap;
        update();
        mseTmp = mse(orthoTmp, type);
        if(type == CEM_MSE){
          b(1, 0) = mseTmp;
        }
        else{
          b(1, 0) = orthoTmp;
        }


        Linalg<TPrecision>::AddScale(Zswap, -2*s, sync, Z);
        update();
        mseTmp = mse(orthoTmp, type);
        if(type == CEM_MSE){
          b(2, 0) = mseTmp;
        }
        else{
          b(2, 0) = orthoTmp;
        }
;

        if(verbose > 1){
          std::cout << "line search: " << std::endl;
          std::cout << b(0, 0) << std::endl;
          std::cout << b(1, 0) << std::endl;
          std::cout << b(2, 0) << std::endl;
        }

        A(0, 2) = 1;
        A(1, 0) = 1*s*s;
        A(1, 1) = -1*s;
        A(1, 2) = 1;
        A(2, 0) = 4*s*s;
        A(2, 1) = -2*s;
        A(2, 2) = 1;

        DenseMatrix<TPrecision> q = Linalg<TPrecision>::Solve(A, b);

        //do step
        if( q(0, 0) > 0){
          TPrecision h = -q(1, 0)/(2*q(0, 0));
          if(h < -2*s){
            h = -2*s;
          }
          else if( h > 1){
            h = 1;
          }
          Linalg<TPrecision>::AddScale(Ztmp, h, sync, Z);
        }
        else if( b(0,0) > b(1, 0) ){
          //do nothing step to -10*s
        }
        else{
          //stop gradient descent - no step
          Zswap = Ztmp;
          Ztmp = Z;
          Z = Zswap;
          //Linalg<TPrecision>::AddScale(Ztmp, -s, sync, Z);
        }

        A.deallocate();
        b.deallocate();
        q.deallocate();

        



        update();

        TPrecision obj = mse(ortho, type); 
        if(verbose > 0){
          std::cout << "Iteration: " << i << std::endl;
          std::cout << "MSE: " <<  obj << std::endl;     
          std::cout << "Ortho: " <<  ortho << std::endl << std::endl;
        }
 
        bool stop = false;
        if(type == CEM_MSE){ 
           stop = objPrev <= obj;
        }
        else{
          stop = orthoPrev <= ortho;
        }
        if(stop){
          Zswap = Ztmp;
          Ztmp = Z;
          Z = Zswap;
          break;
        }
        
        objPrev = obj;      
        orthoPrev = ortho;

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
          //leave one out
          //if(i==index) continue;
          //int nn = KNNY(i, index);
          TPrecision w = KY(i, index);
          Linalg<TPrecision>::AddScale(out, w, Z, i, out);
          sumw += w;
        }     
        Linalg<TPrecision>::Scale(out, 1.f/sumw, out);
      };




      //f(x) - reconstruction mapping
      void f( DenseVector<TPrecision> &y, Vector<TPrecision> &out){
        Linalg<TPrecision>::Zero(out);
        TPrecision sumw = 0;
        for(unsigned int i=0; i < Y.N(); i++){
          TPrecision w = kernelY[i].f(y);
          //TPrecision w = kY.f(y, Y, i);
          Linalg<TPrecision>::AddScale(out, w, Z, i, out);
          sumw += w;
        }     
        Linalg<TPrecision>::Scale(out, 1.f/sumw, out);
      };


      /*
      //---g 0-order

      //g(y_index) - reconstruction mapping
      void g(unsigned int index, Vector<TPrecision> &out){
      Linalg<TPrecision>::Set(out, 0);

      TPrecision sum = 0;

      computefY();


      for(unsigned int i=0; i < knnX; i++){
      int j = KNNX(i, index);
      double w = kernelX.f(fY, j, fY, index);
      Linalg<TPrecision>::AddScale(out, w, Y, j, out); 
      sum += w;
      }

      Linalg<TPrecision>::Scale(out, 1.f/sum, out);
      };


      //g(x) - reconstruction mapping
      void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
      Linalg<TPrecision>::Set(out, 0);

      computefY();

      TPrecision sum = 0;
      for(unsigned int i=0; i < Y.N(); i++){ 
      TPrecision w = kernelX.f(x, fY, i);
      Linalg<TPrecision>::AddScale(out, w, Y, i, out); 
      sum += w;
      }
      Linalg<TPrecision>::Scale(out, 1.f/sum, out);
      };

       */




      //------g 1-order
      //g(x_index) - reconstruction mapping
      void g(unsigned int index, Vector<TPrecision> &out){
        DenseMatrix<TPrecision> sol = LeastSquares(index);

        for(unsigned int i=0; i<Y.M(); i++){
          out(i) = sol(0, i);
        }

        sol.deallocate();
      };  



      //g(x_index) - reconstruction mapping plus tangent plane
      void g(unsigned int index, Vector<TPrecision> &out, Matrix<TPrecision> &J){
        DenseMatrix<TPrecision> sol = LeastSquares(index);

        for(unsigned int i=0; i<Y.M(); i++){
          out(i) = sol(0, i);
        }
        for(unsigned int i=0; i< Z.M(); i++){
          for(unsigned int j=0; j< Y.M(); j++){
            J(j, i) = sol(1+i, j);
          }
        }

        sol.deallocate();
      };




      //g(x) - reconstruction mapping
      void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
        DenseMatrix<TPrecision> sol = LeastSquares(x);

        for(unsigned int i=0; i<Y.M(); i++){
          out(i) = sol(0, i);
        }

        sol.deallocate();
      };   




      //g(x) - reconstruction mapping + tangent plane
      void g( Vector<TPrecision> &x, Vector<TPrecision> &out, Matrix<TPrecision> &J){
        DenseMatrix<TPrecision> sol = LeastSquares(x);
        for(unsigned int i=0; i<Y.M(); i++){
          out(i) = sol(0, i);
        }     
        for(unsigned int i=0; i< Z.M(); i++){
          for(unsigned int j=0; j< Y.M(); j++){
            J(j, i) = sol(1+i, j);
          }
        }

        sol.deallocate();
      };




      //get original data
      DenseMatrix<TPrecision> getY(){
        return Y;
      };



      //get Z (parameters for f)
      DenseMatrix<TPrecision> getZ(){
        return Z;
      };


      //coordinate mapping of Ypoints
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


      DenseMatrix<TPrecision> reconstruct(){
        DenseMatrix<TPrecision> proj(Y.M(), Y.N());
        DenseVector<TPrecision> yp(Y.M()); 
        for(unsigned int i=0; i < Y.N(); i++){
          g(i, yp);
          Linalg<TPrecision>::SetColumn(proj, i, yp);
        }
        yp.deallocate();
        return proj;
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
          DenseMatrix<TPrecision> Xt,
          DenseVector<TPrecision> &p, DenseVector<TPrecision> &var,
          DenseVector<TPrecision> &pk, bool useDensity){

        //update fY if necessary

        DenseMatrix<TPrecision> Yp = reconstruct(fY);

        TPrecision cod = (TPrecision) (Y.M() - fY.M())/2.0;

        //compute trainig set squared distances
        DenseVector<TPrecision> sdist(Y.N());
        for(unsigned int i=0; i< Y.N(); i++){
          sdist(i) = sl2metric.distance(Y, i, Yp, i);
        }

        DenseVector<TPrecision> k(Y.N());

        //compute variances and pdf values
        TPrecision c = -cod * log(2*M_PI);
        DenseVector<TPrecision> xt(Xt.M());

        for(unsigned int i=0; i < Xt.N(); i++){

          TPrecision sum = 0;
          TPrecision vartmp = 0;
          for(unsigned int j=0; j < fY.N(); j++){
            k(j) = kernelX.f(Xt, i, fY, j);
            sum += k(j);
            vartmp += sdist(j) * k(j); 
          } 
          var(i) = vartmp / sum;

          TPrecision d = sl2metric.distance(Yt, i, Ytp, i);
          p(i) = c - cod * log(var(i)) - d / ( 2 * var(i) ) ;
        }

        if(useDensity){
          TPrecision n = log(fY.N());
          KernelDensity<TPrecision> kd(fY, kernelX);
          for(unsigned int i=0; i<p.N(); i++){
            pk(i) = log( kd.p(Xt, i, true)) - n;
          }
        }


        xt.deallocate(); 
        sdist.deallocate(); 
        k.deallocate();
        Yp.deallocate();
      };




      TPrecision getSigmaX(){
        return kernelX.getKernelParam();
      };



      GaussianKernel<TPrecision> getKernelX(){
        return kernelX;
      };




      MahalanobisKernel<TPrecision> *getKernelsY(){
        return kernelY;
      };



      MahalanobisKernel<TPrecision> *estimateKernelsY(DenseMatrix<TPrecision> &data){
        unsigned int N = data.N();
        MahalanobisKernel<TPrecision> *kY = new MahalanobisKernel<TPrecision>[N];
        DenseVector<TPrecision> x(Z.M());
        for(int i = 0; i<N; i++){
          DenseMatrix<TPrecision> ev(Y.M(), Z.M());
          DenseVector<TPrecision> vars(Z.M());
          DenseVector<TPrecision> mean(Y.M());
          Linalg<TPrecision>::ExtractColumn(data, i, x);
          TPrecision varOrtho = varDecomposition(x, ev, vars, mean);
          MahalanobisKernelParam<TPrecision> param(ev, vars, varOrtho, mean);
          kY[i].setKernelParam(param);
        }
        x.deallocate();
        return kY;
      };	






      private:


      void init(){
        kernelX = GaussianKernel<TPrecision>( Z.M());
        fY = Linalg<TPrecision>::Copy(Z);

        unsigned int N = Y.N();

        if(knnX <= Z.M()){
          knnX = Z.M()+1;
        }	  
        if(knnX >= N){
          knnX = N-1;
        }
        if(knnY > N){
          knnY = N;
        }

        KY = DenseMatrix<TPrecision>(N, N);
        sumKY = DenseVector<TPrecision>(N);
        Linalg<TPrecision>::Set(sumKY, 0);
        KYN = DenseMatrix<TPrecision>(N, N);
        KNNY =  DenseMatrix<int>(knnY, N);
        KNNYD = DenseMatrix<TPrecision>(knnY, N);
        Geometry<TPrecision>::computeKNN(Y, KNNY, KNNYD, sl2metric);

        KNNX = DenseMatrix<int>(knnX+1, N);
        KNNXD = DenseMatrix<TPrecision>(knnX+1, N);
        kernelX = GaussianKernel<TPrecision>( Z.M());
        KX = DenseMatrix<TPrecision>(N, N);
        sumKX = DenseVector<TPrecision>(N);
        Linalg<TPrecision>::Set(sumKX, 0);
        KXN = DenseMatrix<TPrecision>(N, N);


      };     





      //update coordinate mapping and nearest neighbors
      void update(){
        computefY();
        updateKNNX();
      };





      //compute cooridnates of trainign data fY = f(Y)
      void computefY(){
        DenseVector<TPrecision> tmp(Z.M());
        for(unsigned int i=0; i<Y.N(); i++){
          f(i, tmp);
          Linalg<TPrecision>::SetColumn(fY, i, tmp);
        }
        tmp.deallocate();
      };






      //update nearest nieghbors of f(Y) for faster gradient computation
      void updateKNNX(){
        unsigned int N = Y.N();
        Geometry<TPrecision>::computeKNN(fY, KNNX, KNNXD, sl2metric);
        for(unsigned int i=0; i < N; i++){
          sumKX(i) = 0;
          for(unsigned int j=0; j < N; j++){
            KX(j, i) = kernelX.f(fY, j, fY, i); 
            sumKX(i) += KX(j, i);
          }
        }

        for(unsigned int i=0; i < KX.M(); i++){
          for(unsigned int j=0; j< KX.N(); j++){
            KXN(i, j) = KX(i, j) / sumKX(j); 
          }
        }
      }; 





      //Kernel bandwidth for manifold mapping -  non-adaptive isotropic
      void computeKernelX(TPrecision alpha){
        TPrecision sigma = 0;
        sX = 0;
        for(unsigned int i=0; i < Z.N(); i++){
          sigma += sqrt( KNNXD(knnX, i) );
        }
        sigma *= alpha/Z.N();
        sX = sigma;

        kernelX.setKernelParam(sigma);
      };




      //Inital kernel bandwidth estimate for coordinate mapping non-adaptive isotropic  based on knnY distances
      void initKY(){
        unsigned int N = Y.N();
        kernelY = new MahalanobisKernel<TPrecision>[N];
        for(int i = 0; i<N; i++){
          DenseMatrix<TPrecision> ev(Y.M(), Z.M());
          DenseVector<TPrecision> vars(Z.M());
          DenseVector<TPrecision> mean = Linalg<TPrecision>::ExtractColumn(Y, i);
          MahalanobisKernelParam<TPrecision> param(ev, vars, 1, mean);
          kernelY[i].setKernelParam(param);
        }

        TPrecision sigma = 0;
        for(unsigned int i=0; i<N; i++){
          sigma += sqrt( KNNYD(knnY-1, i) ); 
        }
        sigma /= N;
        kY = GaussianKernel<TPrecision>(sigma, Z.M());
        for(unsigned int i=0; i < N; i++){
          sumKY(i) = 0;
          for(unsigned int j=0; j < N; j++){
            KY(j, i) = kY.f(Y, j, Y, i);
            sumKY(i) += KY(j, i);
          }
        }

        for(unsigned int i=0; i < KY.M(); i++){
          for(unsigned int j=0; j< KY.N(); j++){
            KYN(i, j) = KY(i, j) / sumKY(j); 
          }
        }

      };




      //Variance decomposition into variance within tangent and normal to the
      //tangent based on knn
      TPrecision varDecomposition(int index, DenseMatrix<TPrecision> ev,
          DenseVector<TPrecision> var, DenseVector<TPrecision> mean){
        TPrecision varO=0;
        TPrecision sumw = 0;
        TPrecision sumw2 = 0;
        int knn=knnY;
        if(knn > knnX){
          knn = knnX;
        }
        for(int i=0; i<knn; i++){
          int nn = KNNX(i, index);
          TPrecision w = 1;//KX(nn, index);
          g(nn, mean);
          varO += w* sl2metric.distance(Y, nn, mean);
          sumw += w;
          sumw2 += w*w;
        }
        TPrecision nw = sumw / (sumw*sumw - sumw2);
        varO = varO *nw;

        g(index, mean, ev);
        Linalg<TPrecision>::QR_inplace(ev);


        DenseVector<TPrecision> diff(Y.M());
        DenseVector<TPrecision> lPlane(var.N());
        Linalg<TPrecision>::Zero(var);
        for(int i=0; i<knn; i++){
          int nn = KNNX(i, index);
          //TPrecision w = KX(nn, index);
          Linalg<TPrecision>::Subtract(Y, nn, mean, diff);
          Linalg<TPrecision>::Multiply(ev, diff, lPlane, true);
          for(int j=0; j<var.N(); j++){
            var(j) += lPlane(j) *lPlane(j);
          }
        }
        Linalg<TPrecision>::Scale(var, 1.0/(knn-1.0), var);


        diff.deallocate();
        lPlane.deallocate();

        return varO;
      };


      //Variance decomposition into variance within tangent and normal to the
      //tangent based on knn
      TPrecision varDecomposition(DenseVector<TPrecision> x, DenseMatrix<TPrecision> ev,
          DenseVector<TPrecision> var, DenseVector<TPrecision> mean){      
        int k=knnY;
        if(k > knnX){
          k = knnX;
        }
        DenseVector<int> knn(k);
        DenseVector<TPrecision> knnD(k);
        Geometry<TPrecision>::computeKNN(fY, x, knn, knnD, sl2metric);

        TPrecision varO=0;
        TPrecision sumw = 0;
        TPrecision sumw2 = 0;
        for(int i=0; i<k; i++){
          int nn = knn(i);
          TPrecision w = 1;//KX(nn, index);
          g(nn, mean);
          varO += w* sl2metric.distance(Y, nn, mean);
          sumw += w;
          sumw2 += w*w;
        }
        TPrecision nw = sumw / (sumw*sumw - sumw2);
        varO = varO *nw;

        g(x, mean, ev);
        Linalg<TPrecision>::QR_inplace(ev);


        DenseVector<TPrecision> diff(Y.M());
        DenseVector<TPrecision> lPlane(var.N());
        Linalg<TPrecision>::Zero(var);
        for(int i=0; i<k; i++){
          int nn = knn(i);        
          //TPrecision w = KX(nn, index);
          Linalg<TPrecision>::Subtract(Y, nn, mean, diff);
          Linalg<TPrecision>::Multiply(ev, diff, lPlane, true);
          for(int j=0; j<var.N(); j++){
            var(j) += lPlane(j) *lPlane(j);
          }
        }
        Linalg<TPrecision>::Scale(var, 1.0/(k-1.0), var);


        diff.deallocate();
        lPlane.deallocate();
        knn.deallocate();
        knnD.deallocate();

        return varO;
      };






      //Update kernel for coordinate mapping (f / lambda)
      //kernel is center at g(f(y)) with bandwidth decomposition estimated 
      //by avergae projection distance and neareast neighbor distance in tangent plane
      void updateKY(bool updateK = true){
        unsigned int N = Y.N();

        if(updateK){
          DenseVector<TPrecision> varPlane(Z.M());
          Linalg<TPrecision>::Zero(varPlane);
          TPrecision varOrtho = 0;
          for(int i=0; i<N; i++){
            MahalanobisKernelParam<TPrecision> &p = kernelY[i].getKernelParam();
            p.varOrtho = varDecomposition(i, p.ev, p.var, p.mean);
            varOrtho += p.varOrtho;
            Linalg<TPrecision>::Add(varPlane, p.var, varPlane);
          } 

          std::cout << "vo: " << varOrtho/Y.N() << std::endl;
          Linalg<TPrecision>::Scale(varPlane, 1.0/Y.N(), varPlane);
          for(int i=0; i<varPlane.N(); i++){
            std::cout << "vp: " << varPlane(i) << std::endl;
          }
          varPlane.deallocate();


        }

        for(unsigned int i=0; i < N; i++){
          sumKY(i) = 0;
          for(unsigned int j=0; j < N; j++){
            KY(j, i) = kernelY[i].f(Y, j);
            //KY(j, i) = kY.f(Y, j, Y, i);
            sumKY(i) += KY(j, i);
          }
        }

        for(unsigned int i=0; i < KY.M(); i++){
          for(unsigned int j=0; j< KY.N(); j++){
            KYN(i, j) = KY(i, j) / sumKY(j); 
          }
        }

      }










      //numerical gradient at point r of the training data
      //epsilon = finite difference delta
      void numGradX(int r, DenseVector<TPrecision> &gx, TPrecision epsilon, int type){

        TPrecision eg = 0;
        TPrecision og = 0;
        //TPrecision e = mse(r);
        TPrecision o = 0;
        TPrecision  e = mse(r, o, type);
        //vary each coordinate
        for(unsigned int i=0; i<gx.N(); i++){
          Z(i, r) += epsilon;
          numGradLocalUpdate(r);
          //eg = mse(r);
          eg = mse(r, og, type);
          if(type == CEM_MSE){ 
            gx(i) = ( eg - e ) / epsilon;
          }
          else{
            gx(i) = ( og - o ) / epsilon;
          }
          Z(i, r) -= epsilon;
        }

        //update nearest neighbors
        numGradLocalUpdate(r);

      };


      TPrecision mse(int index, TPrecision &o, int type){
        o=0;
        TPrecision e = 0;

        //Jacobian of g(x)
        DenseMatrix<TPrecision> J(Y.M(), Z.M());

        //Temp vars.
        DenseVector<TPrecision> gfy(Y.M());
        DenseVector<TPrecision> diff(Y.M());
        DenseVector<TPrecision> pDot(Z.M());

        int knn = std::min(knnX, knnY);
        for(int nt=0; nt<1; nt++){
          for(unsigned int i=0; i < knn; i++){
            int nn = 0;
            if(nt == 0){
              nn = KNNX(i, index);
            }
            else{
              nn= KNNY(i, index);
            }
            g(nn, gfy, J);
            //e += sl2metric.distance(Y, nn, gfy);

            Linalg<TPrecision>::Subtract(gfy, Y, nn, diff);  
            e += Linalg<TPrecision>::SquaredLength(diff);
            if(type == CEM_ORTHO_NORMALIZE){
              Linalg<TPrecision>::QR_inplace(J);
            }
            if(type == CEM_ORTHO_NORMALIZE2){
              Linalg<TPrecision>::Normalize(diff);
            }
            Linalg<TPrecision>::Multiply(J, diff, pDot, true);

            for(unsigned int n=0; n< pDot.N(); n++){
              o += pDot(n)*pDot(n);
            }  
          }
        }
        o = o/(2*knn);

        pDot.deallocate();
        gfy.deallocate();
        diff.deallocate();
        J.deallocate();

        return e/(2*knn);
      };


      //Local mse
      TPrecision mse(int index){
        TPrecision e = 0;
        DenseVector<TPrecision> gfy(Y.M());
        int knn = std::min(knnX, knnY);
        for(unsigned int i=0; i < knn; i++){
          int nn = KNNX(i, index);
          g(nn, gfy);
          e += sl2metric.distance(Y, nn, gfy); 
        }
        gfy.deallocate();
        return e/knn;
      };




      //update local neighborhood
      void numGradLocalUpdate(int r){
        DenseVector<TPrecision> fy(Z.M());
        int knn = std::min(knnX, knnY);
        for(unsigned int k=0; k<knn; k++){
          int nn = KNNX(k, r);
          f(nn, fy);
          Linalg<TPrecision>::SetColumn(fY, nn, fy);

          nn = KNNY(k, r);
          f(nn, fy);
          Linalg<TPrecision>::SetColumn(fY, nn, fy);
        }

        fy.deallocate();
      };








      //Least squares for locally linear regression at x
      DenseMatrix<TPrecision> LeastSquares(Vector<TPrecision> &x){
        //Compute KNN
        DenseVector<int> knn(knnX);
        DenseVector<TPrecision> knnDist(knnX);
        Geometry<TPrecision>::computeKNN(fY, x, knn, knnDist, sl2metric);

        //Linear system
        DenseMatrix<TPrecision> A(knnX, Z.M()+1);
        DenseMatrix<TPrecision> b(knnX, Y.M());

        //Setup linear system
        for(unsigned int i=0; i < knnX; i++){
          unsigned int nn = knn(i);
          TPrecision w = kernelX.f(knnDist(i));
          A(i, 0) = w;
          for(unsigned int j=0; j< Z.M(); j++){
            A(i, j+1) = (fY(j, nn)-x(j)) * w;
          }

          for(unsigned int m = 0; m<Y.M(); m++){
            b(i, m) = Y(m, nn) *w;
          }
        }

        //Solve system
        DenseMatrix<TPrecision> sol = Linalg<TPrecision>::LeastSquares(A, b);

        //cleanup
        knn.deallocate();
        knnDist.deallocate();
        A.deallocate();
        b.deallocate();
        return sol;
      };





      //Linear least squares for locally linear regression within training data
      DenseMatrix<TPrecision> LeastSquares(unsigned int n){

        //Linear system
        DenseMatrix<TPrecision> A(knnX, Z.M()+1);
        DenseMatrix<TPrecision> b(knnX, Y.M());

        //Setup linear system with pre-computed weights

        for(unsigned int i=0; i < knnX; i++){

          //leave one out
          //unsigned int nn = KNNX(i+1, n);
          unsigned int nn = KNNX(i, n);

          TPrecision w = KX(nn, n);
          A(i, 0) = w;
          for(unsigned int j=0; j< Z.M(); j++){
            A(i, j+1) = (fY(j, nn)-fY(j, n)) * w;
          }

          for(unsigned int m = 0; m<Y.M(); m++){
            b(i, m) = Y(m, nn) *w;
          }
        }

        //Solve linear system
        DenseMatrix<TPrecision> sol = Linalg<TPrecision>::LeastSquares(A, b);

        //cleanup
        A.deallocate();
        b.deallocate();
        return sol;

      };

    }; 


#endif

