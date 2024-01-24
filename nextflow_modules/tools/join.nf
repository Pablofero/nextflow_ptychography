#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python ) 
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.join) // "navigate"/"select" the right section in the Yaml
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

process join {
        publishDir "$params.outputfolder/join/$workflow.runName", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            // tuple path(recon), path(datasets_h5) , path(probe_positions),  path(tile_shape), path(tile_offset), path(slice_no_overlap),path(tile_no_overlab_sub_shape) 
            path recon
            path positions
            path tile_shape
            path tile_offset
            path slice_no_overlap
            path tile_no_overlab_sub_shape
            path total_tiles_shape
            path extra_vacuum_space
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
            path "recon.tiff", emit: recon
            path "recon_colorbar.pdf", emit: recon_colorbar_pdf
            path "recon_for_vis.png", emit: recon_for_vis_png
            path "recon_without_vacuum_norm_colorbar.pdf", emit: recon_without_vacuum_norm_colorbar_pdf
            path "recon_without_vacuum_norm_for_vis.png", emit: recon_without_vacuum_for_vis_png
            path "vmin_vmax.txt", emit: vmin_vmax_txt
            path "*",includeInputs: true, emit: all
            // path "beta_ds_joined.tiff", emit: beta
            // path "delta_ds_joined.tiff", emit: delta
            // path "probe_mag_ds_joined.tiff", emit: probe_mag
            // path "probe_phase_ds_joined.tiff", emit: probe_phase
        // echo true // output standart out to terminal

        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            python $projectDir/bin/mytools/join.py --recon $recon --positions $positions --tile_shape $tile_shape --tile_offset $tile_offset --slice_no_overlap $slice_no_overlap --tile_no_overlab_sub_shape $tile_no_overlab_sub_shape --total_tiles_shape $total_tiles_shape --extra_vacuum_space $extra_vacuum_space $expandedParameters
            """
    }

     
 


