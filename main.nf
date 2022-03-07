#!/usr/bin/env nextflow
nextflow.enable.dsl=2

//cmd: nextflow main.nf -params-file  config.yaml -with-report report.html

//default parameters
params.outputfolder = "output"

//import modules
include {unwarp} from "./nextflow_modules/unwarp_workflow.nf"
include {make_adorym_data} from "./nextflow_modules/make_adorym_data.nf"
include {make_adorym_positions} from "./nextflow_modules/make_adorym_positions.nf"

workflow {
    datasets = channel.fromPath(params.datafolder+'/'+params.datasets)
    ref = channel.fromPath(params.datafolder+'/'+params.unwarp_ref)
    warp = channel.fromPath(params.datafolder+'/'+params.unwarp_warp)
    
    unwarped_datasets = unwarp(datasets, ref, warp)
    make_adorym_data = make_adorym_data(unwarped_datasets)

                datasets_h5 = make_adorym_data.datasets_h5
                beamstop = make_adorym_data.beamstop
                debug_png = make_adorym_data.debug_png


    beam_pos = make_adorym_positions().pos
}



// useful: http://nextflow-io.github.io/patterns/index.html