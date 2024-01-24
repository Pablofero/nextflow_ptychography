#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

process rop_compile { 
    label 'rop_compile'
    input:
      path list

    output:
      path "ROP"

    script:
      """
      nvcc -c *.cu -std=c++14
      nvcc -c *.cpp -std=c++14
      nvcc *.o -lcudart -lcublas -lcufft -lculibos -o ROP # original instruction indicate use of static libs: -lcudart_static -lcublas_static -lcufft_static -lculibos
      """
}