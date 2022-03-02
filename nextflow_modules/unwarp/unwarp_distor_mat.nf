#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.unwarp.unwarp_distor_mat)
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams)

process unwarp_distor_mat {
        label 'all_cpu'
        publishDir params.outputfolder+"/unwarp_distor_mat/", mode: 'copy'
        output:
            path "ab_distortion_matrix.npy", emit: ab_mat // unwarp_distor_mat.out.ab_mat to acces this output chanel
            path "*.txt" optional true //if save_all_precompute is true, then *.txt are created
            path "*.png" optional true
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp/unwarp_distor_mat.py $expandedParameters --cpu_count=$task.cpus
            """
    }