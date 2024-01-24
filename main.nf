#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!! (https://github.com/nextflow-io/nextflow/issues/2736)
nextflow.enable.dsl=2 

//cmd: nextflow run main.nf -resume -params-file config.yaml -name run1 --outputfolder output1 -with-report report.html


//default parameters
params.output_folder = "output_"
params.outputfolder = params.output_folder+"/"+workflow.runName //+'_'+workflow.start+'_'+workflow.runName
params.datafolder =  "./data"
params.datasets = ""
println("run name: "+workflow.runName)
// println("outputfolder: "+params.outputfolder)
// conditionally add subprocesses/workflows
do_unwarp_workflow = false
if(params.find{ it.key == "unwarp_workflow" }){
    include {unwarp_workflow} from "./nextflow_modules/unwarp/unwarp_workflow.nf"
    do_unwarp_workflow = true
}
do_make_data = false
if(params.find{ it.key == "make_data" }){
    include {make_data} from "./nextflow_modules/tools/make_data.nf"
    do_make_data = true
}
do_adorym_workflow = false
if(params.find{ it.key == "adorym_workflow" }){
    include {adorym_workflow} from "./nextflow_modules/adorym/adorym_workflow.nf"
    do_adorym_workflow = true
}
do_rop_workflow = false
if(params.find{ it.key == "rop_workflow" }){
    include {rop_workflow} from "./nextflow_modules/rop/rop_workflow.nf"
    do_rop_workflow = true
}
do_join = false
if(params.find{ it.key == "join" }){
    include {join} from "./nextflow_modules/tools/join.nf"
    do_join = true
}
include {render_dag} from "./nextflow_modules/tools/render_dag.nf"
include {copy_config_to_output_folder} from "./nextflow_modules/tools/copy_config_to_output_folder.nf"

workflow {
    println("outputfolder: "+params.outputfolder)
    //create input channel(s)
    datasets = channel.fromPath(params.datafolder+'/'+params.datasets)
    if(do_unwarp_workflow){
        //create input channel(s)
        ref = channel.fromPath(params.datafolder+'/'+params.unwarp_workflow.unwarp_distor_mat.unwarp_ref)
        warp = channel.fromPath(params.datafolder+'/'+params.unwarp_workflow.unwarp_distor_mat.unwarp_warp)

        datasets = unwarp_workflow(datasets, ref, warp) // call the unwarp subworkflow, were the unwarping matrix is calculated and used to unwarp the data
    }
    if (do_make_data){
        make_data_out = make_data(datasets)
            datasets = make_data_out.datasets_h5.transpose() // by explicitly saving the output of the process we make it appear in the dag visualization
            beamstop = make_data_out.beamstop.first()
            total_tiles_shape = make_data_out.total_tiles_shape
            extra_vacuum_space = make_data_out.extra_vacuum_space
            probe_size = make_data_out.probe_size.first()
            pixel_size = make_data_out.pixel_size.first()
            probe_generated = make_data_out.probe_generated.first()
            debug_png = make_data_out.debug_png

        if(do_adorym_workflow){
            datasets_adorym =  adorym_workflow(datasets, beamstop, probe_size, debug_png)
            join(datasets_adorym.collect{it[0]}, datasets_adorym.collect{it[1]}, datasets_adorym.collect{it[2]}, datasets_adorym.collect{it[3]}, datasets_adorym.collect{it[4]}, datasets_adorym.collect{it[5]},total_tiles_shape,extra_vacuum_space)
        }
        if(do_rop_workflow){
            rop_sourcecode = channel.fromPath("$workflow.projectDir/bin/rop/rop/{*.cu,*.cpp,*.h,*.hpp}").toList()
            datasets_rop = rop_workflow(datasets, probe_size, pixel_size, debug_png, probe_generated, rop_sourcecode)
            join(datasets_rop.collect{it[0]}, datasets_rop.collect{it[1]}, datasets_rop.collect{it[2]}, datasets_rop.collect{it[3]}, datasets_rop.collect{it[4]}, datasets_rop.collect{it[5]},total_tiles_shape,extra_vacuum_space)
        }
    }else{
        if (do_adorym_workflow){
            println("\nno make_data specified so can't run adorym subworkflow!")
        }
        if (do_rop_workflow){
            println("\nno make_data specified so can't run rop subworkflow!")
        }
        // if(do_make)  TODO add warning for do_rop_workflow
    }
}


workflow.onComplete {
    copy_config_to_output_folder(params.outputfolder)
    render_dag(params.outputfolder,'pdf')
}


// useful: http://nextflow-io.github.io/patterns/index.html