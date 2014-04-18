#!/bin/bash -x

install_name_tool -change libkmeansrex.so /Users/emonson/Programming/em_cpp_mapa/lib/KMeansRex/libkmeansrex.so /Users/emonson/Programming/em_cpp_mapa/lib/KMeansRex/KMeansRex.mexmaci64

install_name_tool -change libkmeansrex.so /Users/emonson/Programming/em_cpp_mapa/lib/KMeansRex/libkmeansrex.so /Users/emonson/Programming/em_cpp_mapa/lib/KMeansRex/KMeansSeedsRex.mexmaci64
