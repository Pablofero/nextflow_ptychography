#!/usr/bin/env nextflow
nextflow.enable.dsl=2

//cmd: nextflow main.nf -params-file  config.yaml

//default parameters
params.outputfolder = "output"

//import modules
include {unwarp} from "./nextflow_modules/unwarp_workflow.nf"
include {make_adorym_data} from "./nextflow_modules/make_adorym_data.nf"
include {make_adorym_positions} from "./nextflow_modules/make_adorym_positions.nf"

workflow {
    data = channel.fromPath('/testpool/ops/pablofernandezrobledo/Workflows/nextflow_preprocessing/data/Spectrum Image (Dectris)_100mrad_pelz_unfiltered_*.npy')
    unwarp(data) | make_adorym_data
    make_adorym_positions()
}



// useful: http://nextflow-io.github.io/patterns/index.html