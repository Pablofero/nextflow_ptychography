#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.confFile = 'unwarp.json'

process unwarp_memmap {
        label 'all_cpu'
        input:
            path  file_
        output:
            path "${file_.getName().replaceAll(/.npy/, "_unwarped.npy")}" //"${file(file_).getSimpleName()}_unwarped.npy"
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/unwarp_memmap.py --cfg $projectDir/conf/$params.confFile --f_list [$file_] --cpu_count=$task.cpus
            """
    }