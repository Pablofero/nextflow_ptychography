#!/usr/bin/env nextflow
//use the newer version of nextflow (subworflows,etc see https://www.nextflow.io/docs/latest/dsl2.html) do not put comments after the dls=2 line!!
nextflow.enable.dsl=2

process adorym_no_join {
        publishDir "$params.outputfolder/adorym/", mode: 'copy' // location to save the outputs, copying as the default is to symbolically link!
        input:
            path  recon
        output: // for understanding path: https://www.nextflow.io/docs/latest/process.html?#output-path
                // emit: define a name identifier that can be used to reference the channel in the external scope, see also https://www.nextflow.io/docs/latest/dsl2.html?highlight=emit#process-named-output
             path "$recon/beta_ds_1.tiff", emit: beta
             path "$recon/delta_ds_1.tiff", emit: delta
             path "$recon/probe_mag_ds_1.tiff", emit: probe_mag
             path "$recon/probe_phase_ds_1.tiff", emit: probe_phase
        // echo true // output standart out to terminal

        script: //default Bash, see https://www.nextflow.io/docs/latest/process.html#script
            """
            echo blib
            """
    }