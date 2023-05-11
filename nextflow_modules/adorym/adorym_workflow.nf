#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

// conditionally include modules
params.adorym_workflow.do_make_adorym_data = false
if(params.adorym_workflow.find{ it.key == "make_adorym_data" }){ // Todo detect if its a dic or only has one element, if one element->name of file in params.datafolder?
    include {make_adorym_data} from "./make_adorym_data.nf" 
    params.adorym_workflow.do_make_adorym_data = true
}
params.adorym_workflow.do_make_adorym_reconstruct = false
if(params.adorym_workflow.find{ it.key == "make_adorym_reconstruct" }){
    include {make_adorym_reconstruct} from "./make_adorym_reconstruct.nf"
    params.adorym_workflow.do_make_adorym_reconstruct = true
}
params.adorym_workflow.do_adorym_reconstruct = false
if(params.adorym_workflow.find{ it.key == "adorym_reconstruct" }){
    include {adorym_reconstruct} from "./adorym_reconstruct.nf"
    params.adorym_workflow.do_adorym_reconstruct = true
}
params.adorym_workflow.do_adorym_join = false
if(params.adorym_workflow.find{ it.key == "adorym_join" }){
    include {adorym_join} from "./adorym_join.nf"
    params.adorym_workflow.do_adorym_join = true
}else{
    include {adorym_no_join} from "./adorym_no_join.nf"
}
workflow adorym_workflow {
        take://inputs to the (sub)workflow
            datasets
        main:
            if(params.adorym_workflow.do_make_adorym_data){
                make_adorym_data = make_adorym_data(datasets)

                        datasets = make_adorym_data.datasets_h5.transpose() // by explicitly saving the output of the process we make it appear in the dag visualization
                        beamstop = make_adorym_data.beamstop.first()
                        total_tiles_shape = make_adorym_data.total_tiles_shape
                        rot_angle = make_adorym_data.rot_angle
                        probe_size = make_adorym_data.probe_size.first()
                        debug_png = make_adorym_data.debug_png
                        // probe_positions = make_adorym_data.pos
                        // shape = make_adorym_data.shape

            }else{
                make_adorym_data=datasets
                beamstop = '/None' //Todo
                probe_positions = '/None'
            }
            // if(params.adorym_workflow.do_make_adorym_positions){    
            //     beam_pos = make_adorym_positions().pos
            // }else{
            //     beam_pos = null // possibly read in?
            // }

            if(params.adorym_workflow.do_make_adorym_reconstruct){    
                // make_adorym_reconstruct = make_adorym_reconstruct(datasets,beam_pos,beamstop)
                make_adorym_reconstruct = make_adorym_reconstruct(datasets,beamstop,probe_size)
 
                        datasets = make_adorym_reconstruct.datasets_h5.transpose() // by explicitly saving the output of the process we make it appear in the dag visualization
                        beamstop_mar = make_adorym_reconstruct.beamstop.first() //params.adorym_workflow.do_make_adorym_data ? make_adorym_reconstruct.beamstop : '/None'
                        probe_size_mar = make_adorym_data.probe_size.first()
                        // print(params.adorym_workflow.do_make_adorym_data ?' make_adorym_reconstruct.beamstop' : '/None')
                        py_executable = make_adorym_reconstruct.py_executable
                    
            }else{
                datasets = null 
                beamstop = null 
                py_executable = null 
            }

            if(params.adorym_workflow.do_adorym_reconstruct){
                adorym_reconstruct =  adorym_reconstruct(datasets, beamstop_mar, py_executable)
                        datasets = adorym_reconstruct.datasets_h5.transpose() // make_adorym_reconstruct.datasets_h5.transpose() 
                        adorym_recon = adorym_reconstruct.recon.transpose()
            }else{
                adorym_out_possibly_multiples=null
            }
            
            if(params.adorym_workflow.do_adorym_join){
                adorym_out = adorym_join(datasets.collect(),total_tiles_shape,rot_angle, adorym_recon.toList())
            }else{
                adorym_out=adorym_no_join(adorym_recon)
            }

        //  emit: //output of the (sub)workflow
        //     adorym_out
    }