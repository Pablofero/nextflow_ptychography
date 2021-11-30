#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.confFile = 'make_adorym_positions.json'

process make_adorym_positions {
        publishDir "$params.output/adorym_positions"
        output:
            path "*.npy"
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/make_adorym_positions.py --cfg $projectDir/conf/$params.confFile 
            """
    }