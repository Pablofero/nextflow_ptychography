#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

process recon_rop_to_adorym {
        publishDir "$params.outputfolder/${datasets_bin[0].getName().replaceAll(/.bin/,"")}/rop/", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            tuple path(datasets_bin) , path(probe_positions),  path(shape), path(offset),  path(slice_no_overlap), path(tile_no_overlab_sub_shape)
            // path datasets_in  //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            // path probe_positions
            path probe_size
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
            
            tuple path("recon*", includeInputs: false) , path("beam_pos_*.npy", includeInputs: true),  path("tile_*_shape_pixels.npy", includeInputs: true), path("tile_*_offset.npy", includeInputs: true), path("tile_*_slice_no_overlap.npy", includeInputs: true), path("tile_*_pos_shape.npy", includeInputs: true), emit: datasets_bin
        // echo true // output standart out to terminal

        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            python  $projectDir/bin/rop/recon_rop_to_adorym.py --tile_shape $shape --PotentialImag PotentialImag*.bin --PotentialReal PotentialReal*.bin --probe_size $probe_size  --ProbeImag ProbeImag*.bin --ProbeReal ProbeReal*.bin --positions $probe_positions
            """
    }
