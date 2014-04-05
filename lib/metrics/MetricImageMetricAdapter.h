#ifndef METRICIMAGEMETRICADAPTER_H
#define METRICIMAGEMETRICADAPTER_H

#include "ImageMetric.h"
#include "Metric.h"
#include "DenseVector.h"

#include "ImageVectorConverter.h"
#include "itkImage.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImageRegionConstIterator.h"




template<typename TPrecision, typename TImage>
class MetricImageMetricAdapter : public Metric<TPrecision>, public
                           ImageMetric<TPrecision, TImage>{
  
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;


    MetricImageMetricAdapter(ImageVectorConverter<TImage> &c,
        Metric<TPrecision> &im) : converter(c), metric(im){
    };

    TPrecision distance(Vector<TPrecision> &x1, Vector<TPrecision> &x2){
      return metric.distance(x1, x2);
    };

    TPrecision distance(Matrix<TPrecision> &X, int i1, Matrix<TPrecision> &Y, int i2){
      return metric.distance(X, i1, Y, i2);
    };

    TPrecision distance(Matrix<TPrecision> &X, int i1, Vector<TPrecision> &x2){
      return metric.distance(X, i1, x2);
    };

    TPrecision distance(ImagePointer x1, ImagePointer x2) {
      static DenseVector<Precision> s1(converter.getD());
      static DenseVector<Precision> s2(converter.getD());

      converter.extractVector(x1, s1);
      converter.extractVector(x2, s2);
      return distance(s1, s2);
    };

    
  private:
    ImageVectorConverter<TImage> &converter;
    Metric<TPrecision> &metric;
};


#endif
