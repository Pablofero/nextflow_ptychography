#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.unwarp.unwarp_apply)
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams)

process unwarp_apply {
        cpus moduleParams.cpu_count
        publishDir "$params.outputfolder/${file_.getName().replaceAll(/.npy/,'_unwarped')}/unwarped", mode: 'copy'
        input:
            path file_
            path ab_distortion_matrix
        output:
            path "*_unwarped.npy", emit: unwarped
            path "*_unwarped_json.npy" optional true
            path "*.png" optional true 
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp/unwarp_apply.py $expandedParameters --file $file_ --ab $ab_distortion_matrix
            """
    }