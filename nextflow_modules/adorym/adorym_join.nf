#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python ) 
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.adorym_workflow.adorym_reconstruct) // "navigate"/"select" the right section in the Yaml
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

process adorym_join {
        publishDir "$params.outputfolder/adorym/", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            //tuple path(datasets_h5) , path(probe_positions),  path(shape), path(offset), path(slice_no_overlap)
            path  things
            path  total_tiles_shape
            path  rot_angle
            path  recon
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
             path "beta_ds_joined.tiff", emit: beta
             path "delta_ds_joined.tiff", emit: delta
             path "probe_mag_ds_joined.tiff", emit: probe_mag
             path "probe_phase_ds_joined.tiff", emit: probe_phase
        // echo true // output standart out to terminal

        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            python $projectDir/bin/adorym/adorym_join.py --total_tiles_shape $total_tiles_shape --rot_angle $rot_angle --recon $recon
            """
    }