#!/bin/bash -x

g++ --shared -o libkmeansrex.so KMeansRexCore.cpp -I.. -O3 -DNDEBUG -arch x86_64
