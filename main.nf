#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!! (https://github.com/nextflow-io/nextflow/issues/2736)
nextflow.enable.dsl=2 

//cmd: nextflow run main.nf -resume -params-file config.yaml -name run1 --outputfolder output1 -with-report report.html


//default parameters
params.output_folder = "output_"
params.outputfolder = params.output_folder+'_'+workflow.start+'_'+workflow.runName
params.datafolder =  "./data"
params.datasets = ""
println("output_"+workflow.runName)
// conditionally add subprocesses/workflows
do_unwarp_workflow = false
if(params.find{ it.key == "unwarp_workflow" }){
    include {unwarp_workflow} from "./nextflow_modules/unwarp/unwarp_workflow.nf"
    do_unwarp_workflow = true
}

do_adorym_workflow = false
if(params.find{ it.key == "adorym_workflow" }){
    include {adorym_workflow} from "./nextflow_modules/adorym/adorym_workflow.nf"
    do_adorym_workflow = true
}

workflow {
    println(params.outputfolder)
    //create input channel(s)
    datasets = channel.fromPath(params.datafolder+'/'+params.datasets)
    if(do_unwarp_workflow){
        //create input channel(s)
        ref = channel.fromPath(params.datafolder+'/'+params.unwarp_workflow.unwarp_distor_mat.unwarp_ref)
        warp = channel.fromPath(params.datafolder+'/'+params.unwarp_workflow.unwarp_distor_mat.unwarp_warp)

        datasets = unwarp_workflow(datasets, ref, warp) // call the unwarp subworkflow, were the unwarping matrix is calculated and used to unwarp the data
    }

    if(do_adorym_workflow){
        adorym_workflow(datasets)
    }
    
}

include {render_dag} from "./nextflow_modules/tools/render_dag.nf"
include {copy_config_to_output_folder} from "./nextflow_modules/tools/copy_config_to_output_folder.nf"
workflow.onComplete {
    render_dag(params.outputfolder,'pdf')
    copy_config_to_output_folder(params.outputfolder)
}


// useful: http://nextflow-io.github.io/patterns/index.html