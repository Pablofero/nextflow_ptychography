#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.confFile = 'unwarp_apply.json'
params.output = "output"

process unwarp_apply {
        label 'all_cpu'
        publishDir "$params.output/${file_.getName().replaceAll(/.npy/,'_unwarped')}/unwarped", mode: 'copy'
        input:
            path file_
            path ab_distorsion_matrix
        output:
            path "*_unwarped.npy" 
            path "*_unwarped_json.npy" optional true
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp/unwarp_apply.py --cfg $projectDir/conf/$params.confFile --file $file_ --ab $ab_distorsion_matrix --cpu_count=$task.cpus
            """
    }