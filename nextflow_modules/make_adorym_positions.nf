#!/usr/bin/env nextflow
nextflow.enable.dsl=2
include { toArgs1 } from "./tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.make_adorym_positions)
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams)
process make_adorym_positions {
        publishDir "$params.outputfolder/adorym_positions", mode: 'copy'
        output:
            path "*.npy"
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/make_adorym_positions.py $expandedParameters
            """
    }