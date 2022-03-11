#!/usr/bin/env nextflow
nextflow.enable.dsl=2 //use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html)


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python ) 
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.unwarp.unwarp_distor_mat) // "navigate"/"select" the right section in the Yaml 
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

process unwarp_distor_mat {
        cpus moduleParams.cpu_count // extract the cpuy count for configuring slurm/other executor
        publishDir params.outputfolder+"/unwarp_distor_mat/", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            path ref //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            path warp
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
            path "ab_distortion_matrix.npy", emit: ab_mat // unwarp_distor_mat.out.ab_mat to acces this output chanel
            path "*.txt", emit: debug_txt optional true //if save_all_precompute is true, then *.txt are created
            path "*.png", emit: debug_png optional true //if save_all_precompute is true, then *.png 
        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp/unwarp_distor_mat.py $expandedParameters --unwarp_ref $ref --unwarp_warp $warp
            """
    }