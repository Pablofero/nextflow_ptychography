#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

// conditionally include modules
params.adorym_workflow.do_make_adorym_reconstruct = false
if(params.adorym_workflow.find{ it.key == "make_adorym_reconstruct" }){
    include {make_adorym_reconstruct} from "./make_adorym_reconstruct.nf"
    params.adorym_workflow.do_make_adorym_reconstruct = true
}
params.adorym_workflow.do_adorym_reconstruct = false
if(params.adorym_workflow.find{ it.key == "adorym_reconstruct" }){
    include {adorym_reconstruct} from "./adorym_reconstruct.nf"
    params.adorym_workflow.do_adorym_reconstruct = true
}
params.adorym_workflow.do_adorym_join = false

workflow adorym_workflow {
        take://inputs to the (sub)workflow
            datasets
            beamstop
            probe_size
            debug_png
        main:
            if(params.adorym_workflow.do_make_adorym_reconstruct){    
                // make_adorym_reconstruct = make_adorym_reconstruct(datasets,beam_pos,beamstop)
                make_adorym_reconstruct = make_adorym_reconstruct(datasets,beamstop,probe_size)
 
                        datasets = make_adorym_reconstruct.datasets_h5.transpose() // by explicitly saving the output of the process we make it appear in the dag visualization
                        beamstop_mar = make_adorym_reconstruct.beamstop.first() //params.adorym_workflow.do_make_adorym_data ? make_adorym_reconstruct.beamstop : '/None'
                        // probe_size_mar = make_adorym_data.probe_size.first()
                        // print(params.adorym_workflow.do_make_adorym_data ?' make_adorym_reconstruct.beamstop' : '/None')
                        py_executable = make_adorym_reconstruct.py_executable
                    
                if(params.adorym_workflow.do_adorym_reconstruct){
                    adorym_reconstruct =  adorym_reconstruct(datasets, beamstop_mar, py_executable)
                            datasets = adorym_reconstruct.datasets_h5 // make_adorym_reconstruct.datasets_h5.transpose()
                }
            }

         emit: //output of the (sub)workflow
            datasets
    }