#ifndef CEMCONDITIONAL_H
#define CEMCONDITIONAL_H

#include "Vector.h"
#include "Matrix.h"

template <typename TPrecision>
class CEMConditional{

  virtual void evaluate(Vector<TPrecision> &x, Vector<TPrecision> &fx,
      Matrix<TPrecision> &gx, DenseMatrix<TPRecision> &Y,
      DenseMatrxi<TPrecision> &X) = 0;
  
  virtual void evaluate(Vector<TPrecision> &x, Vector<TPrecision> &fx,
      DenseMatrix<TPRecision> &Y, DenseMatrxi<TPrecision> &X) = 0;
  
  virtual void evaluate(Vector<TPrecision> &x, Matrix<TPrecision> &gx,
      DenseMatrix<TPRecision> &Y, DenseMatrxi<TPrecision> &X) = 0;
};

#endif
