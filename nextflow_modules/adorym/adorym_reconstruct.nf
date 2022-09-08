#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python ) 
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.adorym_workflow.adorym_reconstruct) // "navigate"/"select" the right section in the Yaml
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

process adorym_reconstruct {
        clusterOptions "--gres=gpu:1" //custom Slurm settings 
        publishDir "$params.outputfolder/${datasets_h5_in.getName().replaceAll(/.npy/,"")}/adorym/", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            path datasets_h5_in  //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            path probe_positions
            path beamstop
            path py_executable
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
             path "*", emit: adorym_out
            // path "*.h5" , includeInputs: true ,emit: datasets_h5
            // path "beam_pos_*.npy" , includeInputs: true ,emit: probe_positions
            // path "*beamstop.npy" , includeInputs: true ,emit: beamstop
            // path "adorym_recostruct.py" , emit: py_executable
        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            print(task.workDir)
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $py_executable
            """
    }