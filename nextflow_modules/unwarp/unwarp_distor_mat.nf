#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.unwarp.unwarp_distor_mat)
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams)

process unwarp_distor_mat {
        cpus moduleParams.cpu_count
        publishDir params.outputfolder+"/unwarp_distor_mat/", mode: 'copy'
        input:
            path ref
            path warp
        output:
            path "ab_distortion_matrix.npy", emit: ab_mat // unwarp_distor_mat.out.ab_mat to acces this output chanel
            path "*.txt", emit: debug_txt optional true //if save_all_precompute is true, then *.txt are created
            path "*.png", emit: debug_png optional true //if save_all_precompute is true, then *.png 
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp/unwarp_distor_mat.py $expandedParameters --unwarp_ref $ref --unwarp_warp $warp
            """
    }