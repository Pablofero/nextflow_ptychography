#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.confFile_Distor_mat = 'unwarp_distor_mat.json'
params.output = "output"
//import modules after
include {unwarp_distor_mat as distor_mat} from "./unwarp/unwarp_distor_mat.nf" 
include {unwarp_apply} from "./unwarp/unwarp_apply.nf"  addParams(confFile: "unwarp_apply.json")

workflow unwarp {
        take:
            file_
        main:
            distor_mat()
            unwarp_apply(file_,distor_mat.out.ab_mat)
        emit:
            unwarp_apply.out.unwarped
            // path "${file_.getName().replaceAll(/.npy/, "_unwarped.npy")}" //"${file(file_).getSimpleName()}_unwarped.npy"
    }