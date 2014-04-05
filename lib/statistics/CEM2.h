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


//Conditional Expectation Manifolds
template <typename TPrecision>
class CEM{

  private:    
    //high dimensional data
    DenseMatrix<TPrecision> Y;
    //parameters for lambda (coordinate) mapping
    DenseMatrix<TPrecision> L;
    //parameters for g (manifold) mapping
    DenseMatrix<TPrecision> G;

    //lambda(Y) 
    DenseMatrix<TPrecision> lY;

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
    MahalanobisKernel<TPrecision> *kernelY;


    TPrecision sX;



  public:

    virtual void cleanup(){      
      partialCleanup();
      Y.deallocate();
      G.deallocate();
      L.deallocate();
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
      lY.deallocate();
      delete[] kernelY;
    };


    //Create Condtional Expectation Manifold 
    CEM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Linit, 
	TPrecision alpha, unsigned int nnY, unsigned int nnX) :
      Y(Ydata), L(Linit), knnY(nnY), knnX(nnX){


	std::cout << "moo" << std::endl;
	init();
	std::cout << "moo1" << std::endl;
	initKY();
	std::cout << "moo2" << std::endl;
	updateLambda();
	std::cout << "moo3" << std::endl;
	updateG();
	std::cout << "moo4" << std::endl;
	computeKernelX(alpha);
	std::cout << "moo5" << std::endl;
	updateG();
	updateKY();
	std::cout << "moo6" << std::endl;
	updateLambda();
	std::cout << "moo7" << std::endl;
      };



    CEM(DenseMatrix<TPrecision> Ydata, DenseMatrix<TPrecision> Lopt, unsigned
	int nnY, unsigned int nnX, double sigmaX, MahalanobisKernel<TPrecision>
	*kY): Y(Ydata), L(Lopt), knnY(nnY), knnX(nnX), kernelY(kY) {

      init();
      kernelX.setKernelParam(sigmaX);
      sX = kernelX.getKernelParam();  
      updateKY(false);
      updateLambda();
      updateG();

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
    };


    //evalue objective function, squared error
    virtual TPrecision mse(TPrecision &o){
      o=0;
      TPrecision e = 0;

      //Jacobian of g(x)
      DenseMatrix<TPrecision> J(Y.M(), L.M());

      //Temp vars.
      DenseVector<TPrecision> gfy(Y.M());
      DenseVector<TPrecision> diff(Y.M());
      DenseVector<TPrecision> pDot(L.M());

      for(unsigned int i=0; i < Y.N(); i++){
	g(i, gfy, J);
	e += sl2metric.distance(Y, i, gfy);

	Linalg<TPrecision>::Subtract(gfy, Y, i, diff);  

	for(unsigned int n=0; n< J.N(); n++){
	  TPrecision norm = 0;
	  for(unsigned int j=0; j<J.M(); j++){
	    norm += J(j, n) * J(j, n);
	  }
	  if(norm == 0){
	    norm = 0.0001;
	  };
	  norm = sqrt(norm);

	  for(unsigned int j=0; j<J.M(); j++){
	    J(j, n) /= norm;
	  }
	}

	Linalg<TPrecision>::Normalize(diff);
	Linalg<TPrecision>::Multiply(J, diff, pDot, true);

	for(unsigned int n=0; n< pDot.N(); n++){
	  o += acos(sqrt(pDot(n)*pDot(n)));
	}  
      }
      o = o/(L.M()*L.N())/ M_PI * 180;

      pDot.deallocate();
      gfy.deallocate();
      diff.deallocate();
      J.deallocate();


      return e/Y.N();
    };







    //Gradient descent for all points 
    void gradDescent(unsigned int nIterations, TPrecision scaling, int verbose=1){


      //---Storage for syncronous updates 
      DenseMatrix<TPrecision> sync(L.M(), L.N());

      //---Do nIterations of gradient descent     
      DenseMatrix<TPrecision> Ltmp(L.M(), L.N());
      DenseMatrix<TPrecision> Lswap;

      //gradient direction
      DenseVector<TPrecision> gx(L.M());     
      if(verbose > 0){
	std::cout << "Start Gradient Descent" << std::endl << std::endl;
      }

      for(unsigned int outer=0; outer < nIterations; outer++){
	updateKY();
	updateLambda();
	TPrecision orthoPrev =0;
	TPrecision ortho;
	TPrecision objPrev = mse(orthoPrev);
	if(verbose > 0){
	  std::cout << "Mse start: " << objPrev << std::endl;
	  std::cout << "Ortho start: " << orthoPrev << std::endl;
	}

	std::cout << "Update G iterations " << outer << std::endl;
	for(unsigned int i=0; i<4; i++){
	  //compute gradient for each point
	  TPrecision maxL = 0;
	  for(unsigned int j=0; j < L.N(); j++){
	    //compute gradient
	    //gradX(j, gx);
	    numGradX(j, gx, sX/10.f);

	    //store gradient for syncronous updates
	    TPrecision l = Linalg<TPrecision>::Length(gx);
	    if(maxL < l){
	      maxL = l;
	    }
	    //Linalg<TPrecision>::Scale(gx, 1.f/l, gx);
	    for(unsigned int k=0; k<L.M(); k++){
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

	  b(0, 0) = mse();
	  Linalg<TPrecision>::AddScale(L, -1*s, sync, Ltmp);
	  Lswap = L;
	  L = Ltmp;
	  Ltmp = Lswap;
	  updateLambda();

	  b(1, 0) = mse();
	  Linalg<TPrecision>::AddScale(Lswap, -2*s, sync, L);
	  updateLambda();

	  b(2, 0) = mse();

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
	    Linalg<TPrecision>::AddScale(Ltmp, h, sync, L);
	  }
	  else if( b(0,0) > b(1, 0) ){
	    //do nothing step to -10*s
	  }
	  else{
	    //stop gradient descent - no step
	    Lswap = Ltmp;
	    Ltmp = L;
	    L = Lswap;
	    //Linalg<TPrecision>::AddScale(Ltmp, -s, sync, L);
	  }

	  A.deallocate();
	  b.deallocate();
	  q.deallocate();


	  updateLambda();

	  TPrecision obj = mse(ortho); 
	  if(verbose > 0){
	    std::cout << "Iteration: " << i << std::endl;
	    std::cout << "MSE: " <<  obj << std::endl;     
	    std::cout << "Ortho: " <<  ortho << std::endl << std::endl;
	  }   
	  if(objPrev <= obj){// || orthoPrev >= ortho){
	    break;
	  }
	  objPrev = obj;      

	  }
	  updateG();
	  TPrecision obj = mse(ortho); 
	  std::cout << "Upate G MSE: " <<  obj << std::endl;     
	  std::cout << "Update G Ortho: " <<  ortho << std::endl << std::endl;
	}


	//cleanup 
	sync.deallocate(); 
	gx.deallocate();
	Ltmp.deallocate();
      };





      //f(x_index) - reconstruction mapping
      void f(unsigned int index, Vector<TPrecision> &out){
	Linalg<TPrecision>::Zero(out);
	TPrecision sumw = 0;
	for(unsigned int i=0; i < Y.N(); i++){
	  //if(i==index) continue;
	  //int nn = KNNY(i, index);
	  TPrecision w = KY(index, i);
	  Linalg<TPrecision>::AddScale(out, w, L, i, out);
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
	  Linalg<TPrecision>::AddScale(out, w, L, i, out);
	  sumw += w;
	}     
	Linalg<TPrecision>::Scale(out, 1.f/sumw, out);
      };



      //---g 0-order
      /*
      //g(y_index) - reconstruction mapping
      void g(unsigned int index, Vector<TPrecision> &out){
      Linalg<TPrecision>::Set(out, 0);

      TPrecision sum = 0;

      computefY();


      for(unsigned int i=0; i < 2*knnSigma; i++){
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
	DenseVector<TPrecision> x = Linalg<TPrecision>::ExtractColumn(lY, index);
	g(x, out);
      };  



      //g(x_index) - reconstruction mapping plus tangent plane
      void g(unsigned int index, Vector<TPrecision> &out, Matrix<TPrecision> &J){
	DenseVector<TPrecision> x = Linalg<TPrecision>::ExtractColumn(lY, index);
	g(x, out, J);
      };




      //g(x) - reconstruction mapping
      void g( Vector<TPrecision> &x, Vector<TPrecision> &out){
	DenseMatrix<TPrecision> sol = LeastSquares(x);

	for(unsigned int i=0; i<Y.M(); i++){
	  out(i) = sol(0, i);
	}

	sol.deallocate();
      };   




      //g(x) - reconstruction mapping + tangent plance
      void g( Vector<TPrecision> &x, Vector<TPrecision> &out, Matrix<TPrecision> &J){
	DenseMatrix<TPrecision> sol = LeastSquares(x);
	for(unsigned int i=0; i<Y.M(); i++){
	  out(i) = sol(0, i);
	}     
	for(unsigned int i=0; i< L.M(); i++){
	  for(unsigned int j=0; j< Y.M(); j++){
	    J(j, i) = sol(1+i, j);
	  }
	}

	sol.deallocate();
      };





      //get original Y's
      DenseMatrix<TPrecision> getY(){
	return Y;
      };

      //get original Y's
      DenseMatrix<TPrecision> getG(){
	return G;
      };


      //get L (parameters for f
      DenseMatrix<TPrecision> getZ(){
	return L;
      };


      //coordinate mapping fo Ypoints
      DenseMatrix<TPrecision> parametrize(DenseMatrix<TPrecision> &Ypoints){

	DenseMatrix<TPrecision> proj(L.M(), Ypoints.N());
	parametrize(Ypoints, proj);

	return proj;
      };




      //
      void parametrize(DenseMatrix<TPrecision> &Ypoints, DenseMatrix<TPrecision> &proj){

	DenseVector<TPrecision> tmp(Y.M()); 
	DenseVector<TPrecision> xp(L.M()); 

	for(unsigned int i=0; i < Ypoints.N(); i++){
	  Linalg<TPrecision>::ExtractColumn(Ypoints, i, tmp);
	  f(tmp, xp);
	  Linalg<TPrecision>::SetColumn(proj, i, xp);
	}
	xp.deallocate();
	tmp.deallocate();
      };




      DenseMatrix<TPrecision> &parametrize(){
	return lY;
      };



      DenseMatrix<TPrecision> reconstruct(DenseMatrix<TPrecision> &Xpoints){
	DenseMatrix<TPrecision> proj(Y.M(), Xpoints.N());
	reconstruct(Xpoints, proj);     
	return proj;
      };





      void reconstruct(DenseMatrix<TPrecision> &Xpoints, DenseMatrix<TPrecision> &proj){
	DenseVector<TPrecision> tmp(L.M()); 
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

	DenseMatrix<TPrecision> Yp = reconstruct(lY);

	TPrecision cod = (TPrecision) (Y.M() - lY.M())/2.0;

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
	  for(unsigned int j=0; j < lY.N(); j++){
	    k(j) = kernelX.f(Xt, i, lY, j);
	    sum += k(j);
	    vartmp += sdist(j) * k(j); 
	  } 
	  var(i) = vartmp / sum;

	  TPrecision d = sl2metric.distance(Yt, i, Ytp, i);
	  p(i) = c - cod * log(var(i)) - d / ( 2 * var(i) ) ;
	}

	if(useDensity){
	  TPrecision n = log(lY.N());
	  KernelDensity<TPrecision> kd(lY, kernelX);
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










      private:


      void init(){
	kernelX = GaussianKernel<TPrecision>( L.M());
	lY = Linalg<TPrecision>::Copy(L);
	G = Linalg<TPrecision>::Copy(L);

	unsigned int N = Y.N();

	if(knnX <= L.M()){
	  knnX = L.M()+1;
	}	  
	if(knnX > N){
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
	kernelX = GaussianKernel<TPrecision>( L.M());
	KX = DenseMatrix<TPrecision>(N, N);
	sumKX = DenseVector<TPrecision>(N);
	Linalg<TPrecision>::Set(sumKX, 0);
	KXN = DenseMatrix<TPrecision>(N, N);


      };     



      void updateG(){
	Linalg<TPrecision>::Copy(lY, G);
	updateKNNX();

      };



      void updateLambda(){
	DenseVector<TPrecision> tmp(L.M());
	for(unsigned int i=0; i<Y.N(); i++){
	  f(i, tmp);
	  Linalg<TPrecision>::SetColumn(lY, i, tmp);
	}
	tmp.deallocate();
      };






      void updateKNNX(){
	unsigned int N = Y.N();
	Geometry<TPrecision>::computeKNN(G, KNNX, KNNXD, sl2metric);
	for(unsigned int i=0; i < N; i++){
	  sumKX(i) = 0;
	  for(unsigned int j=0; j < N; j++){
	    KX(j, i) = kernelX.f(G, j, G, i); 
	    sumKX(i) += KX(j, i);
	  }
	}

	for(unsigned int i=0; i < KX.M(); i++){
	  for(unsigned int j=0; j< KX.N(); j++){
	    KXN(i, j) = KX(i, j) / sumKX(j); 
	  }
	}
      }; 





      void computeKernelX(TPrecision alpha){
	TPrecision sigma = 0;
	sX = 0;
	for(unsigned int i=0; i < L.N(); i++){
	  sigma += sqrt( KNNXD(knnX, i) );
	}
	sigma *= alpha/L.N();
	sX = sigma;


	//std::cout << "sigmaX: " << sigma << std::endl;
	//std::cout << "scale: " << sX << std::endl;

	kernelX.setKernelParam(sigma);
      };




      void initKY(){
	unsigned int N = Y.N();
	kernelY = new MahalanobisKernel<TPrecision>[N];
	for(int i = 0; i<N; i++){
	  DenseMatrix<TPrecision> ev(Y.M(), L.M());
	  DenseVector<TPrecision> vars(L.M());
	  DenseVector<TPrecision> mean = Linalg<TPrecision>::ExtractColumn(Y, i);
	  MahalanobisKernelParam<TPrecision> param(ev, vars, 1, mean);
	  kernelY[i].setKernelParam(param);
	}

	TPrecision sigma = 0;
	for(unsigned int i=0; i<N; i++){
	  sigma += sqrt( KNNYD(knnY-1, i) ); 
	}
	sigma /= 3*N;
	GaussianKernel<TPrecision> kY = GaussianKernel<TPrecision>(sigma, L.M());
	for(unsigned int i=0; i < N; i++){
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



      TPrecision varDecomposition(int index, DenseMatrix<TPrecision> ev,
	  DenseVector<TPrecision> var, DenseVector<TPrecision> mean){
	TPrecision varO=0;
	TPrecision sumw = 0;
	TPrecision sumw2 = 0;
	for(int i=0; i<knnX; i++){
	  int nn = KNNX(i, index);
	  TPrecision w = KX(nn, index);
	  g(nn, mean);
	  varO += w* sl2metric.distance(Y, nn, mean);
	  sumw += w;
	  sumw2 += w*w;
	}
	TPrecision nw = sumw / (sumw*sumw - sumw2);
	varO = varO *nw;

	g(index, mean, ev);
	DenseMatrix<TPrecision> tmp = Linalg<TPrecision>::QR(ev);
	Linalg<TPrecision>::Copy(tmp, ev);
	tmp.deallocate();
	//DenseVector<TPrecision> ls = Linalg<TPrecision>::ColumnwiseNorm(ev);
	//for(int i=0; i<ev.N(); i++){
	//  Linalg<TPrecision>::ScaleColumn(ev, i, 1.0/ls(i)); 
	//}


	DenseVector<TPrecision> diff(Y.M());
	DenseVector<TPrecision> lPlane(var.N());
	Linalg<TPrecision>::Zero(var);
	for(int i=0; i<knnY; i++){
	  int nn = KNNY(i, index);
	  //TPrecision w = KX(nn, index);
	  Linalg<TPrecision>::Subtract(Y, nn, mean, diff);
	  Linalg<TPrecision>::Multiply(ev, diff, lPlane, true);
	  for(int j=0; j<var.N(); j++){
	    var(j) += lPlane(j) *lPlane(j);
	  }
	}
	Linalg<TPrecision>::Scale(var, 1.0/(knnY-1), var);


	diff.deallocate();
	lPlane.deallocate();
	//ls.deallocate();


	return varO;
      };



      void updateKY(bool updateK = true){
	unsigned int N = Y.N();

	if(updateK){
	  DenseVector<TPrecision> varPlane(L.M());
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
	  for(unsigned int j=0; j < N; j++){
	    KY(j, i) = kernelY[i].f(Y, j);
	    sumKY(i) += KY(j, i);
	  }
	}

	for(unsigned int i=0; i < KY.M(); i++){
	  for(unsigned int j=0; j< KY.N(); j++){
	    KYN(i, j) = KY(i, j) / sumKY(j); 
	  }
	}

      }








      TPrecision mse(int index){
	TPrecision e = 0;
	DenseVector<TPrecision> gfy(Y.M());
	for(unsigned int i=0; i < knnX; i++){
	  int nn = KNNX(i, index);
	  g(nn, gfy);
	  e += sl2metric.distance(Y, nn, gfy); 
	}
	gfy.deallocate();
	return e/knnX;
      };




      //numerical gradient computation for lambda
      void numGradLocalUpdate(int r){
	DenseVector<TPrecision> fy(L.M());
	for(unsigned int k=0; k<knnX; k++){
	  int nn = KNNX(k, r);
	  f(nn, fy);
	  Linalg<TPrecision>::SetColumn(lY, nn, fy);
	}
	/*for(int k=0; k<knnX; k++){
	  int nn = KNNX(k, r);
	  updateKY(nn);
	  }
	  for(unsigned int k=0; k<knnX; k++){
	  int nn = KNNX(k, r);
	  f(nn, fy);
	  Linalg<TPrecision>::SetColumn(fY, nn, fy);
	  }*/
	fy.deallocate();
      };

      virtual void numGradX(int r, DenseVector<TPrecision> &gx, TPrecision epsilon){

	TPrecision eg = 0;
	TPrecision e = mse(r);
	for(unsigned int i=0; i<gx.N(); i++){
	  L(i, r) += epsilon;
	  //update nearest neighbors
	  //f(r, fy);
	  ///Linalg<TPrecision>::SetColumn(fY, r, fy);
	  numGradLocalUpdate(r);
	  eg = mse(r);
	  gx(i) = ( eg - e ) / epsilon;
	  L(i, r) -= epsilon;
	}

	//f(r, fy);
	//Linalg<TPrecision>::SetColumn(fY, r, fy);

	//update nearest neighbors
	numGradLocalUpdate(r);

      };






      //Least squares for g
      DenseMatrix<TPrecision> LeastSquares(Vector<TPrecision> &x){
	DenseVector<int> knn(knnX);
	DenseVector<TPrecision> knnDist(knnX);
	Geometry<TPrecision>::computeKNN(G, x, knn, knnDist, sl2metric);

	DenseMatrix<TPrecision> A(knnX, L.M()+1);
	DenseMatrix<TPrecision> b(knnX, Y.M());

	for(unsigned int i=0; i < knnX; i++){
	  unsigned int nn = knn(i);
	  TPrecision w = kernelX.f(x, G, nn);
	  A(i, 0) = w;
	  for(unsigned int j=0; j< L.M(); j++){
	    A(i, j+1) = (G(j, nn)-x(j)) * w;
	  }

	  for(unsigned int m = 0; m<Y.M(); m++){
	    b(i, m) = Y(m, nn) *w;
	  }
	}

	DenseMatrix<TPrecision> sol = Linalg<TPrecision>::LeastSquares(A, b);

	A.deallocate();
	b.deallocate();
	return sol;
      };











    }; 


#endif

