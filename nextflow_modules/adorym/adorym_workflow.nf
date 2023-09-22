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
if(params.adorym_workflow.find{ it.key == "adorym_join" }){
    include {adorym_join} from "./adorym_join.nf"
    params.adorym_workflow.do_adorym_join = true
}else{
    include {adorym_no_join} from "./adorym_no_join.nf"
}
workflow adorym_workflow {
        take://inputs to the (sub)workflow
            datasets
            beamstop
            total_tiles_shape
            extra_vacuum_space
            probe_size
            debug_png
        main:
            // probe_positions = make_adorym_data.pos
            // shape = make_adorym_data.shape


            // if(params.adorym_workflow.do_make_adorym_positions){    
            //     beam_pos = make_adorym_positions().pos
            // }else{
            //     beam_pos = null // possibly read in?
            // }

            if(params.adorym_workflow.do_make_adorym_reconstruct){    
                // make_adorym_reconstruct = make_adorym_reconstruct(datasets,beam_pos,beamstop)
                make_adorym_reconstruct = make_adorym_reconstruct(datasets,beamstop,probe_size)
 
                        datasets = make_adorym_reconstruct.datasets_h5.transpose() // by explicitly saving the output of the process we make it appear in the dag visualization
                        beamstop_mar = make_adorym_reconstruct.beamstop.first() //params.adorym_workflow.do_make_adorym_data ? make_adorym_reconstruct.beamstop : '/None'
                        // probe_size_mar = make_adorym_data.probe_size.first()
                        // print(params.adorym_workflow.do_make_adorym_data ?' make_adorym_reconstruct.beamstop' : '/None')
                        py_executable = make_adorym_reconstruct.py_executable
                    
            }else{
                datasets = null 
                beamstop = null 
                py_executable = null 
            }

            if(params.adorym_workflow.do_adorym_reconstruct){
                adorym_reconstruct =  adorym_reconstruct(datasets, beamstop_mar, py_executable)
                        datasets = adorym_reconstruct.datasets_h5 // make_adorym_reconstruct.datasets_h5.transpose() 
            }else{
                adorym_out_possibly_multiples=null
            }
            
            if(params.adorym_workflow.do_adorym_join){
                adorym_out = adorym_join(datasets.collect{it[0]}, datasets.collect{it[2]}, datasets.collect{it[3]}, datasets.collect{it[4]}, datasets.collect{it[5]}, datasets.collect{it[6]},total_tiles_shape,extra_vacuum_space) //0 -> recon, 2->probe_positions, 3->tile_shape, 4->tile_offset, 5->slice_no_overlap, 6->tile_no_overlab_sub_shape
                        // beta = adorym_out.beta
                        // delta = adorym_out.delta
                        // probes_mag = adorym_out.probe_mag
                        // probes_phase = adorym_out.probe_phase
            }else{
                //adorym_out=adorym_no_join(adorym_recon)
            }

         emit: //output of the (sub)workflow
            adorym_out.recon
    }