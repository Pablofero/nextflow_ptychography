#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

include {rop_compile} from "./rop_compile.nf"
include {data_adorym_to_rop} from "./data_adorym_to_rop.nf" 
// conditionally include modules
params.rop_workflow.do_make_rop_reconstruct = false
if(params.rop_workflow.find{ it.key == "make_rop_reconstruct" }){
    include {make_rop_reconstruct} from "./make_rop_reconstruct.nf" 
    params.rop_workflow.do_make_rop_reconstruct = true
    
    if(params.rop_workflow.find{ it.key == "rop_reconstruct" }){ 
        include {rop_reconstruct} from "./rop_reconstruct.nf" 
        include {recon_rop_to_adorym} from "./recon_rop_to_adorym.nf" 
        params.rop_workflow.do_rop_reconstruct = true
    }

}

workflow rop_workflow {
        take://inputs to the (sub)workflow
            datasets
            probe_size
            pixel_size
            debug_png
            probe_generated
            sourcecode
        main:
            
            ROP = rop_compile(sourcecode)

            datasets = data_adorym_to_rop(datasets,probe_size)

            if(params.rop_workflow.do_make_rop_reconstruct){    
               
               make_rop_reconstruct_out = make_rop_reconstruct(datasets,probe_generated,probe_size,pixel_size)
                        datasets = make_rop_reconstruct_out.datasets_bin.transpose()
               
               if (rop_reconstruct){

                    datasets_bin = rop_reconstruct(datasets,ROP).datasets_bin
                    datasets = recon_rop_to_adorym(datasets_bin,probe_size)

               }              
            }
            // if(params.rop_workflow.do_rop_reconstruct){
            //     rop_reconstruct =  rop_reconstruct(datasets, beamstop_mar, py_executable)
            //             datasets = rop_reconstruct.datasets_h5 // make_rop_reconstruct.datasets_h5.transpose() 
            // }else{
            //     rop_out_possibly_multiples=null
            // }
            
            // if(params.rop_workflow.do_rop_join){
            //     rop_out = rop_join(datasets.collect{it[0]}, datasets.collect{it[2]}, datasets.collect{it[3]}, datasets.collect{it[4]}, datasets.collect{it[5]}, datasets.collect{it[6]},total_tiles_shape,extra_vacuum_space) //0 -> recon, 2->probe_positions, 3->tile_shape, 4->tile_offset, 5->slice_no_overlap, 6->tile_no_overlab_sub_shape
            //             // beta = rop_out.beta
            //             // delta = rop_out.delta
            //             // probes_mag = rop_out.probe_mag
            //             // probes_phase = rop_out.probe_phase
            // }else{
            //     //rop_out=rop_no_join(rop_recon)
            // }

         emit: //output of the (sub)workflow
            datasets
    }