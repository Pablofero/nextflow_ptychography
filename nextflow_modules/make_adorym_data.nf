#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { toArgs1 } from "./tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.make_adorym_data)
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams)

process make_adorym_data {
        publishDir "$params.outputfolder/${file_.getName().replaceAll(/.npy/,"")}/adorym_data", mode: 'copy'
        input:
            path file_
        output:
            path "*.h5" , emit: datasets_h5
            path "*_beamstop.npy", emit: beamstop
            path "*.png", emit: debug_png optional true
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/make_adorym_data.py $expandedParameters --Path_2_Unwarped $file_
            """
    }