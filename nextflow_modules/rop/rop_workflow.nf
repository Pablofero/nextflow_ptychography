#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

// conditionally include modules
params.rop_workflow.do_make_adorym_data = false
if(params.rop_workflow.find{ it.key == "make_adorym_data" }){ // Todo detect if its a dic or only has one element, if one element->name of file in params.datafolder?
    include {make_adorym_data} from "./make_adorym_data.nf" 
    params.rop_workflow.do_make_adorym_data = true
}

workflow rop_workflow {
        take://inputs to the (sub)workflow
            datasets
        main:
            if(params.rop_workflow.do_make_adorym_data){
                make_adorym_data = make_adorym_data(datasets)

                        datasets = make_adorym_data.datasets_h5.transpose() // by explicitly saving the output of the process we make it appear in the dag visualization
                        beamstop = make_adorym_data.beamstop.first()
                        total_tiles_shape = make_adorym_data.total_tiles_shape
                        extra_vacuum_space = make_adorym_data.extra_vacuum_space
                        probe_size = make_adorym_data.probe_size.first()
                        debug_png = make_adorym_data.debug_png
                        // probe_positions = make_adorym_data.pos
                        // shape = make_adorym_data.shape

            }else{
                make_adorym_data=datasets
                beamstop = '/None' //Todo
                probe_positions = '/None'
            }
            // if(params.rop_workflow.do_make_adorym_positions){    
            //     beam_pos = make_adorym_positions().pos
            // }else{
            //     beam_pos = null // possibly read in?
            // }

            if(params.rop_workflow.do_make_adorym_reconstruct){    
                // make_adorym_reconstruct = make_adorym_reconstruct(datasets,beam_pos,beamstop)
                make_adorym_reconstruct = make_adorym_reconstruct(datasets,beamstop,probe_size)
 
                        datasets = make_adorym_reconstruct.datasets_h5.transpose() // by explicitly saving the output of the process we make it appear in the dag visualization
                        beamstop_mar = make_adorym_reconstruct.beamstop.first() //params.rop_workflow.do_make_adorym_data ? make_adorym_reconstruct.beamstop : '/None'
                        probe_size_mar = make_adorym_data.probe_size.first()
                        // print(params.rop_workflow.do_make_adorym_data ?' make_adorym_reconstruct.beamstop' : '/None')
                        py_executable = make_adorym_reconstruct.py_executable
                    
            }else{
                datasets = null 
                beamstop = null 
                py_executable = null 
            }

            if(params.rop_workflow.do_adorym_reconstruct){
                adorym_reconstruct =  adorym_reconstruct(datasets, beamstop_mar, py_executable)
                        datasets = adorym_reconstruct.datasets_h5 // make_adorym_reconstruct.datasets_h5.transpose() 
            }else{
                adorym_out_possibly_multiples=null
            }
            
            if(params.rop_workflow.do_adorym_join){
                adorym_out = adorym_join(datasets.collect{it[0]}, datasets.collect{it[2]}, datasets.collect{it[3]}, datasets.collect{it[4]}, datasets.collect{it[5]}, datasets.collect{it[6]},total_tiles_shape,extra_vacuum_space) //0 -> recon, 2->probe_positions, 3->tile_shape, 4->tile_offset, 5->slice_no_overlap, 6->tile_no_overlab_sub_shape
                        // beta = adorym_out.beta
                        // delta = adorym_out.delta
                        // probes_mag = adorym_out.probe_mag
                        // probes_phase = adorym_out.probe_phase
            }else{
                //adorym_out=adorym_no_join(adorym_recon)
            }

        //  emit: //output of the (sub)workflow
        //     adorym_out
    }