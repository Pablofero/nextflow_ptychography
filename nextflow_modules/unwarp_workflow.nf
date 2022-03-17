#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

//import modules
include {unwarp_distor_mat} from "./unwarp/unwarp_distor_mat.nf" 
include {unwarp_apply} from "./unwarp/unwarp_apply.nf"

workflow unwarp {
        take://inputs to the (sub)workflow
            datasets
            ref
            warp
        main:
            unwarp_distor_mat = unwarp_distor_mat(ref,warp) // calculate the  distortion matrix
            ab_mat = unwarp_distor_mat.ab_mat
                    debug_txt = unwarp_distor_mat.debug_txt // by explicitly saving the output of the process we make it appear in the dag visualization
                    debug_png = unwarp_distor_mat.debug_png

            unwarp_apply = unwarp_apply(datasets,ab_mat.first())
                    unwarped = unwarp_apply.unwarped
                    debug_png = unwarp_apply.debug_png

        emit: //output of the (sub)workflow
            unwarped
    }