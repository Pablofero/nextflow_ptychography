#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.confFile = 'make_adorym_data.json'

process make_adorym_data {
        publishDir "$params.outputfolder/${file_.getName().replaceAll(/.npy/,"")}/adorym_data", mode: 'copy'
        input:
            path file_
        output:
            path "*.h5" 
            path "*_beamstop.npy"
            path "*.png" optional true
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/make_adorym_data.py --cfg $projectDir/conf/$params.confFile --Path2Unwarped $file_
            """
    }