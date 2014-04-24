#ifndef IMAGEMETRIC_H
#define IMAGEMETRIC_H

#include <math.h>

template<typename TPrecision, typename TImage>
class ImageMetric{
    
  public:
    typedef TImage Image;
    typedef typename Image::Pointer ImagePointer;

    virtual TPrecision distance(ImagePointer x1, ImagePointer x2) = 0;
    
    virtual ImagePointer computeDistanceImage(ImagePointer x1, ImagePointer x2){
      return NULL;
    };
};


#endif
