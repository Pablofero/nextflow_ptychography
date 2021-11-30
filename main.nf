#!/usr/bin/env nextflow
nextflow.enable.dsl=2

//paramteres
params.output = "output"
//import modules
// include {unwarp_memmap as unwarp} from "./nextflow_modules/unwarp.nf" addParams(confFile: "unwarp.json")
include {unwarp} from "./nextflow_modules/unwarp_workflow.nf" addParams(confFile_Distor_mat: "unwarp_distor_mat.json")
include {make_adorym_data} from "./nextflow_modules/make_adorym_data.nf" addParams(confFile: "make_adorym_data.json")
include {make_adorym_positions} from "./nextflow_modules/make_adorym_positions.nf" addParams(confFile: "make_adorym_positions.json")



workflow {
    data = channel.fromPath('/testpool/ops/pablofernandezrobledo/Workflows/nextflow_preporcessing/data/Spectrum Image (Dectris)_100mrad_pelz_unfiltered_*.npy')
    unwarp(data) | make_adorym_data
    make_adorym_positions()
    // println("\n\nview:\n")
    // make_adorym_data_out.view()
}

// usefull: http://nextflow-io.github.io/patterns/index.html

/*notes:
using collect to processes that need all data at once.
*/