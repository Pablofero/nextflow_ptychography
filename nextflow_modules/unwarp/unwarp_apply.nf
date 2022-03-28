#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python ) 
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.unwarp_workflow.unwarp_apply) // "navigate"/"select" the right section in the Yaml 
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams)// create a string with the parameters given in the yaml in teh format: --key "value"

process unwarp_apply {
        cpus moduleParams.cpu_count  // extract the cpuy count for configuring slurm/other executor
        publishDir "$params.outputfolder/${file_.getName().replaceAll(/.npy/,'_unwarped')}/unwarped", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            path file_ //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            path ab_distortion_matrix
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
            path "*_unwarped.npy", emit: unwarped
            path "*.png", emit: debug_png optional true  
        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp/unwarp_apply.py $expandedParameters --file $file_ --ab $ab_distortion_matrix
            """
    }