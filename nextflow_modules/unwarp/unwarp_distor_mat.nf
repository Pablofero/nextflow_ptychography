#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.confFile = 'unwarp_distor_mat.json'
params.output = "output"

process unwarp_distor_mat {
        label 'all_cpu'
        publishDir params.output+"/unwarp_distor_mat/", mode: 'copy'
        output:
            path "ab_distorsion_matrix.npy", emit: ab_mat // unwarp_distor_mat.out.ab_mat to acces this output chanel
            path "*.txt" optional true //if save_all_precompute is true, then *.txt are created
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp/unwarp_distor_mat.py --cfg $projectDir/conf/$params.confFile --cpu_count=$task.cpus
            """
    }