#ifndef IMAGEMETRICMETRICADAPTER_H
#define IMAGEMETRICMETRICADAPTER_H

#include "ImageMetric.h"
#include "Metric.h"

#include "ImageVectorConverter.h"
#include "itkImage.h"
#include "itkGradientMagnitudeImageFilter.h"
#include "itkImageRegionConstIterator.h"




template<typename TPrecision, typename TImage>
class ImageMetricMetricAdapter : public Metric<TPrecision>, public
                           ImageMetric<TPrecision, TImage>{
  
  
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;


    ImageMetricMetricAdapter(ImageVectorConverter<TImage> &c,
        ImageMetric<TPrecision, TImage> &im) : converter(c), metric(im){
      i1 = converter.createImage();
      i2 = converter.createImage();
    };

    virtual TPrecision distance(Vector<TPrecision> &x1, Vector<TPrecision> &x2){
      converter.fillImage(x1, i1);
      converter.fillImage(x2, i2);
      return metric.distance(i1, i2);
    };


    virtual TPrecision distance(Matrix<TPrecision> &X, int ix, Vector<TPrecision> &x2){
      converter.fillImage(x2, i2);
      converter.fillImage(X, ix, i1);
      return metric.distance(i1, i2);
    };

    virtual TPrecision distance(Matrix<TPrecision> &X, int ix,
        Matrix<TPrecision> &Y, int iy){
      converter.fillImage(X, ix, i1);
      converter.fillImage(Y, iy, i2);
      return metric.distance(i1, i2);
    };

    TPrecision distance(ImagePointer x1, ImagePointer x2){
      return metric.distance(x1, x2);
    };

    
  private:
    ImagePointer i1;
    ImagePointer i2;


    ImageVectorConverter<Image> &converter;
    ImageMetric<TPrecision, TImage> &metric;
};


#endif
