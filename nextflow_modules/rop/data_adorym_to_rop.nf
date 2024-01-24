#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python )
include { toArgs1 } from "../tools/toArgs1.nf"

process data_adorym_to_rop {
        input:
            tuple path(datasets_h5) , path(probe_positions),  path(shape), path(offset), path(slice_no_overlap), path(tile_no_overlab_sub_shape)   //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            path probe_size
        output:
            tuple path("*.bin", includeInputs: false) , path("beam_pos_*.npy", includeInputs: true),  path("tile_*_shape_pixels.npy", includeInputs: true), path("tile_*_offset.npy", includeInputs: true), path("tile_*_slice_no_overlap.npy", includeInputs: true), path("tile_*_pos_shape.npy", includeInputs: true), path("CBEDDim.npy", includeInputs: true) 
        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            python $projectDir/bin/rop/data_adorym_to_rop.py --data $datasets_h5 --position $probe_positions --probe_size $probe_size
            """
    }
    //path("beam_pos_*_rop_*.npy", includeInputs: true)