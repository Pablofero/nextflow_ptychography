#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2 

//cmd: nextflow main.nf -params-file  config.yaml -with-report report.html

//default parameters
params.outputfolder = "output"

//import modules
include {unwarp} from "./nextflow_modules/unwarp_workflow.nf"
include {make_adorym_data} from "./nextflow_modules/make_adorym_data.nf"
include {make_adorym_positions} from "./nextflow_modules/make_adorym_positions.nf"

workflow {
    //create input channels
    datasets = channel.fromPath(params.datafolder+'/'+params.datasets)
    ref = channel.fromPath(params.datafolder+'/'+params.unwarp_ref)
    warp = channel.fromPath(params.datafolder+'/'+params.unwarp_warp)
    
    unwarped_datasets = unwarp(datasets, ref, warp) // call the unwarp subworkflow, were the unwarping matrix is calculated and used to unwarp the data
    make_adorym_data = make_adorym_data(unwarped_datasets)

                datasets_h5 = make_adorym_data.datasets_h5 // by explicitly saving the output of the process we make it appear in the dag visualization
                beamstop = make_adorym_data.beamstop
                debug_png = make_adorym_data.debug_png


    beam_pos = make_adorym_positions().pos
}



// useful: http://nextflow-io.github.io/patterns/index.html