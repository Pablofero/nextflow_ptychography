#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python ) 
include { toArgs1 } from "./tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.make_adorym_data) // "navigate"/"select" the right section in the Yaml
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

process make_adorym_data {
        publishDir "$params.outputfolder/${file_.getName().replaceAll(/.npy/,"")}/adorym_data", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            path file_ //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
            path "*.h5" , emit: datasets_h5
            path "*_beamstop.npy", emit: beamstop
            path "*.png", emit: debug_png optional true
        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/make_adorym_data.py $expandedParameters --Path_2_Unwarped $file_
            """
    }