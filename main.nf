#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!! (https://github.com/nextflow-io/nextflow/issues/2736)
nextflow.enable.dsl=2 

//cmd: nextflow run main.nf -resume config.yaml -name run1 --outputfolder output1 -with-report report.html


//default parameters
params.outputfolder = "output"
datafolder: "./data"


// conditionally add subprocesses/workflows
do_unwarp_workflow = false
if(params.find{ it.key == "unwarp_workflow" }){
    include {unwarp_workflow} from "./nextflow_modules/unwarp_workflow.nf"
    do_unwarp_workflow = true
}
do_make_adorym_data = false
if(params.find{ it.key == "make_adorym_data" }){
    include {make_adorym_data} from "./nextflow_modules/make_adorym_data.nf"
    do_make_adorym_data = true
}
do_make_adorym_positions = false
if(params.find{ it.key == "make_adorym_positions" }){
    include {make_adorym_positions} from "./nextflow_modules/make_adorym_positions.nf"
    do_make_adorym_positions = true
}


workflow {
    //create input channel(s)
    datasets = channel.fromPath(params.datafolder+'/'+params.datasets)
    
    if(do_unwarp_workflow){
        //create input channel(s)
        ref = channel.fromPath(params.datafolder+'/'+params.unwarp_workflow.unwarp_distor_mat.unwarp_ref)
        warp = channel.fromPath(params.datafolder+'/'+params.unwarp_workflow.unwarp_distor_mat.unwarp_warp)

        datasets = unwarp_workflow(datasets, ref, warp) // call the unwarp subworkflow, were the unwarping matrix is calculated and used to unwarp the data
    }
    if(do_make_adorym_data){
    make_adorym_data = make_adorym_data(datasets)

                datasets_h5 = make_adorym_data.datasets_h5 // by explicitly saving the output of the process we make it appear in the dag visualization
                beamstop = make_adorym_data.beamstop
                debug_png = make_adorym_data.debug_png
    }
    if(do_make_adorym_positions){    
        beam_pos = make_adorym_positions().pos
    }
    
}

include {render_dag} from "./nextflow_modules/tools/render_dag.nf"
include {copy_config_to_output_folder} from "./nextflow_modules/tools/copy_config_to_output_folder.nf"
workflow.onComplete {
    render_dag(params.outputfolder)
    copy_config_to_output_folder(params.outputfolder)
    print 'done'
}


// useful: http://nextflow-io.github.io/patterns/index.html