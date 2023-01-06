#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2


// extract the corresponding parameters from the params map (java terminology, equivalent to dictionary in python )
include { toArgs1 } from "../tools/toArgs1.nf"
moduleParams= new nextflow.script.ScriptBinding$ParamsMap(params.adorym_workflow.make_adorym_reconstruct) // "navigate"/"select" the right section in the Yaml
// argParams = new nextflow.script.ScriptBinding$ParamsMap(moduleParams.argParams)
expandedParameters = toArgs1(moduleParams) // create a string with the parameters given in the yaml in teh format: --key "value"

// obj_size = "" // if do_make_adorym_data is not been executed assumes obj_size is in adorym_reconstruct parameters
// if(params.adorym_workflow.do_make_adorym_data)
// obj_size = "--obj_size '("+params.adorym_workflow.make_adorym_data.scan_pos_list[0]+","+params.adorym_workflow.make_adorym_data.scan_pos_list[1]+",1)'"

process make_adorym_reconstruct {
        publishDir "$params.outputfolder/${datasets_h5.getName().replaceAll(/.npy/,"")}/adorym", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            tuple path(datasets_h5) , path(probe_positions),  path(shape)  //see https://www.nextflow.io/docs/latest/process.html?#input-of-type-path
            //path datasets_h5
            // path probe_positions
            path beamstop 
            path probe_size
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
            tuple path("*.h5", includeInputs: true) , path("beam_pos_*.npy", includeInputs: true),  path("tile_*_shape.npy", includeInputs: true), emit: datasets_h5
            // path {"*_beamstop.npy","None"} , includeInputs: true , optional: true, emit: beamstop
            path "*_beamstop.npy" , includeInputs: true , optional: true, emit: beamstop
            path "probe_size.npy" , includeInputs: true , emit: probe_size
            path "adorym_recostruct.py" , emit: py_executable
        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            println "$datasets_h5"
            """
            python $projectDir/bin/adorym/make_adorym_reconstruct.py --fname $datasets_h5 --probe_pos $probe_positions --probe_size $probe_size --shape $shape $expandedParameters --output_folder '"recon"' --beamstop $beamstop  # beamstop explanation, if '/None' is passed  do nothing else add the argument --beamstop $beamstop'
            """
    }// $obj_size

//  /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/bin/adorym/make_adorym_reconstruct.py $expandedParameters  --fname $datasets_h5  `if [ "$beamstop" != '/None' ] ; then printf %s '--beamstop $beamstop'; fi`  $obj_size # beamstop explanation, if '/None' is passed do nothjing else add the argument --beamstop $beamstop




// /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/bin/adorym/make_adorym_reconstruct.py $expandedParameters  --fname $datasets_h5  --beamstop $beamstop $obj_size
// --obj_size '(${params.adorym_workflow.make_adorym_data.scan_pos_list[0]},${params.adorym_workflow.make_adorym_data.scan_pos_list[1]},1)'