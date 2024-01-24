#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python )
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.rop_workflow.make_rop_reconstruct) // "navigate"/"select" the right section in the Yaml
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

process make_rop_reconstruct {
        publishDir "$params.outputfolder/${datasets_bin.getName().replaceAll(/.bin/,"")}/rop", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            tuple path(datasets_bin) , path(probe_positions),  path(shape), path(offset), path(slice_no_overlap), path(tile_no_overlab_sub_shape), path(CBEDDim)   //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            path probe_generated
            path probe_size
            path pixel_size
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
            tuple path("_*.bin", includeInputs: true) , path("beam_pos_*.npy", includeInputs: true),  path("tile_*_shape_pixels.npy", includeInputs: true), path("tile_*_offset.npy", includeInputs: true), path("tile_*_slice_no_overlap.npy", includeInputs: true), path("tile_*_pos_shape.npy", includeInputs: true), path("Params.cnf", includeInputs: true), path("CBEDDim.npy", includeInputs: true), path("Probe_re.bin", includeInputs: true), path("Probe_im.bin", includeInputs: true), path("Object_re.bin", includeInputs: true), path("Object_im.bin", includeInputs: true), emit: datasets_bin
        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            python $projectDir/bin/rop/make_rop_reconstruct.py  --ObjectDim $shape  --ProbeDim $probe_size --PixelSize $pixel_size --CBEDDim $CBEDDim --beam_position $probe_positions $expandedParameters --probe_generated $probe_generated
            """
    }