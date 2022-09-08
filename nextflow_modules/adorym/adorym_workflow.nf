#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

// conditionally include modules
do_make_adorym_data = false
if(params.adorym_workflow.find{ it.key == "make_adorym_data" }){ // Todo detect if its a dic or only has one element, if one element->name of file in params.datafolder?
    include {make_adorym_data} from "./make_adorym_data.nf" 
    do_make_adorym_data = true
}
do_make_adorym_positions = false
if(params.adorym_workflow.find{ it.key == "make_adorym_positions" }){ // Todo detect if its a dic or only has one element, if one element->name of file in params.datafolder?
    include {make_adorym_positions} from "./make_adorym_positions.nf"
    do_make_adorym_positions = true
}
do_make_adorym_reconstruct = false
if(params.adorym_workflow.find{ it.key == "make_adorym_reconstruct" }){
    include {make_adorym_reconstruct} from "./make_adorym_reconstruct.nf"
    do_make_adorym_reconstruct = true
}
do_adorym_reconstruct = false
if(params.adorym_workflow.find{ it.key == "adorym_reconstruct" }){
    include {adorym_reconstruct} from "./adorym_reconstruct.nf"
    do_adorym_reconstruct = true
}
workflow adorym_workflow {
        take://inputs to the (sub)workflow
            datasets
        main:
            if(do_make_adorym_data){
                make_adorym_data = make_adorym_data(datasets)

                        datasets = make_adorym_data.datasets_h5 // by explicitly saving the output of the process we make it appear in the dag visualization
                        beamstop = make_adorym_data.beamstop
                        debug_png = make_adorym_data.debug_png

            }else{
                make_adorym_data=null // possibly read in?
            }

            if(do_make_adorym_positions){    
                beam_pos = make_adorym_positions().pos
            }else{
                beam_pos = null // possibly read in?
            }

            if(do_make_adorym_reconstruct){    
                make_adorym_reconstruct = make_adorym_reconstruct(datasets,beam_pos,beamstop)
 
                        datasets = make_adorym_reconstruct.datasets_h5 // by explicitly saving the output of the process we make it appear in the dag visualization
                        probe_positions = make_adorym_reconstruct.probe_positions
                        beamstop = make_adorym_reconstruct.beamstop
                        py_executable = make_adorym_reconstruct.py_executable
            }else{
                datasets = null 
                probe_positions = null 
                beamstop = null 
                py_executable = null 
            }
            if(do_adorym_reconstruct){
                adorym_out = adorym_reconstruct(datasets, probe_positions, beamstop, py_executable).adorym_out
            }else{
                adorym_out=null
            }
         emit: //output of the (sub)workflow
            adorym_out
    }