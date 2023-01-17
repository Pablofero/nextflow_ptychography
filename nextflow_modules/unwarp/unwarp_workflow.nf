#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

// conditionally import modules

do_unwarp_distor_mat = false
if(params.unwarp_workflow.find{ it.key == "unwarp_distor_mat" }){ // Todo detect if its a dic or only has one element, if one element->name of file in params.datafolder?
    include {unwarp_distor_mat} from "./unwarp_distor_mat.nf" 
    do_unwarp_distor_mat = true
}
do_unwarp_apply = false
if(params.unwarp_workflow.find{ it.key == "unwarp_apply" }){ // Todo detect if its a dic or only has one element, if one element->name of file in params.datafolder?
    include {unwarp_apply} from "./unwarp_apply.nf" 
    do_unwarp_apply = true
}



workflow unwarp_workflow {
        take://inputs to the (sub)workflow
            datasets
            ref
            warp
        main:
            if(do_unwarp_distor_mat){
            unwarp_distor_mat = unwarp_distor_mat(ref,warp) // calculate the  distortion matrix
            ab_mat = unwarp_distor_mat.ab_mat
                    debug_txt = unwarp_distor_mat.debug_txt // by explicitly saving the output of the process we make it appear in the dag visualization
                    debug_png = unwarp_distor_mat.debug_png
            }else{
                if(do_unwarp_apply){
                    ab_mat= channel.fromPath(params.datafolder+'/'+params.unwarp_workflow.unwarp_apply.ab_mat)
                }
                ab_mat=Null
            }
            if(do_unwarp_apply){
            unwarp_apply = unwarp_apply(datasets,ab_mat.first())
                    unwarped = unwarp_apply.unwarped
                    debug_png = unwarp_apply.debug_png
            }else{
                unwarped=Null
            }

        emit: //output of the (sub)workflow
            unwarped
    }