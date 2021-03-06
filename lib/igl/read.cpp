// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2013 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "read.h"

#include "readOBJ.h"
#include "readOFF.h"
#include "pathinfo.h"

#include <cstdio>
#include <iostream>


template <typename Scalar, typename Index>
IGL_INLINE bool igl::read(
  const std::string str,
  std::vector<std::vector<Scalar> > & V,
  std::vector<std::vector<Index> > & F)
{
  using namespace std;
  using namespace igl;
  // dirname, basename, extension and filename
  string d,b,e,f;
  pathinfo(str,d,b,e,f);
  // Convert extension to lower case
  std::transform(e.begin(), e.end(), e.begin(), ::tolower);
  vector<vector<Scalar> > TC, N;
  vector<vector<Index> > FTC, FN;
  if(e == "obj")
  {
    return readOBJ(str,V,TC,N,F,FTC,FN);
  }else if(e == "off")
  {
    return readOFF(str,V,F,N);
  }
  cerr<<"Error: "<<__FUNCTION__<<": "<<
    str<<" is not a recognized mesh file format."<<endl;
  return false;
}


#ifndef IGL_NO_EIGN
template <typename DerivedV, typename DerivedF>
IGL_INLINE bool igl::read(
  const std::string str,
  Eigen::PlainObjectBase<DerivedV>& V,
  Eigen::PlainObjectBase<DerivedF>& F)
{
    const char* p;
    for (p = str.c_str(); *p != '\0'; p++)
        ;
    while (*p != '.')
        p--;
    
    if (!strcmp(p, ".obj") || !strcmp(p, ".OBJ"))
    {
        return igl::readOBJ(str,V,F);
    }else if (!strcmp(p, ".off") || !strcmp(p, ".OFF"))
    {
        return igl::readOFF(str,V,F);
    }
    else 
    {
      fprintf(stderr,"read() does not recognize extension: %s\n",p);
      return false;
    }
}
#endif

#ifndef IGL_HEADER_ONLY
// Explicit template specialization
// generated by autoexplicit.sh
template bool igl::read<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
template bool igl::read<double, int>(std::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<std::vector<double, std::allocator<double> >, std::allocator<std::vector<double, std::allocator<double> > > >&, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > >&);
template bool igl::read<Eigen::Matrix<double, -1, 3, 0, -1, 3>, Eigen::Matrix<int, -1, 3, 0, -1, 3> >(std::string, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 3, 0, -1, 3> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, 3, 0, -1, 3> >&);
#endif
