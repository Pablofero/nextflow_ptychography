#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python ) 
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.adorym_workflow.adorym_reconstruct) // "navigate"/"select" the right section in the Yaml
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

process adorym_reconstruct {
        label 'adorym'
        publishDir "$params.outputfolder/${datasets_h5.getName().replaceAll(/.npy/,"")}/adorym/", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            tuple path(datasets_h5) , path(probe_positions),  path(shape)
            // path datasets_h5_in  //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            // path probe_positions
            path beamstop
            path py_executable
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
             path "*", emit: adorym_out
             path "recon*/beta_ds_*.tiff", emit: beta
             path "recon*/delta_ds_*.tiff", emit: delta
             path "recon*/probe_mag_ds_*.tiff", emit: probe_mag
             path "recon*/probe_phase_ds_*.tiff", emit: probe_phase
        // echo true // output standart out to terminal

        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            python $py_executable
            """
    }
    // /opt/anaconda3/envs/tompekin-basic/bin/