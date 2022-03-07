#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.outputfolder = "output"
//import modules after
include {unwarp_distor_mat} from "./unwarp/unwarp_distor_mat.nf" 
include {unwarp_apply} from "./unwarp/unwarp_apply.nf"

workflow unwarp {
        take:
            datasets
            ref
            warp
        main:
            unwarp_distor_mat = unwarp_distor_mat(ref,warp)
            ab_mat = unwarp_distor_mat.ab_mat
                    debug_txt = unwarp_distor_mat.debug_txt
                    debug_png = unwarp_distor_mat.debug_png

            unwarp_apply = unwarp_apply(datasets,ab_mat)
                    unwarped = unwarp_apply.unwarped
                    debug_png = unwarp_apply.debug_png
        emit:
            unwarped
            // path "${file_.getName().replaceAll(/.npy/, "_unwarped.npy")}" //"${file(file_).getSimpleName()}_unwarped.npy"
    }