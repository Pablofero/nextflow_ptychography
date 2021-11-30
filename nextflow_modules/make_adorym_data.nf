#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.confFile = 'make_adorym_data.json'

// import groovy.json.JsonSlurper //from https://groups.google.com/g/nextflow/c/qzsORfO5CFU/m/pYh-tEWXAgAJ & https://github.com/NYU-Molecular-Pathology/demux-nf/blob/a86b6aeeb73933a5e22574aa22772ecf016bb450/main.nf#L9
// def jsonSlurper = new JsonSlurper() // new File object from your JSON file
// def ConfigFile = new File("conf/"+params.confFile)
// String ConfigJSON = ConfigFile.text // load the text from the JSON
// def myConfig = jsonSlurper.parseText(ConfigJSON)// create a dictionary object from the JSON text
// //log.info "Some val: ${myConfig.out_name_append}"// access values in the dict

process make_adorym_data {
                publishDir "$params.output/${file_.getName().replaceAll(/.npy/,"")}/adorym_data", mode: 'copy'
        input:
            path file_
        output:
            // log.info file("${file(file_).getSimpleName()}*.h5")
            path "*.h5" 
            path "*_beamstop.npy"
            // path "${file_.getName().replaceAll(/.npy/,"")}*.h5" 
            // path "${file_.getName().replaceAll(/.npy/,"")}*_beamstop.npy"
            //"${file_.getName().replaceAll(/.npy/, myConfig.out_name_append +'.h5')}" //normaly: "${file(file_).getSimpleName()}_unwarped.h5". but the other works with files with two or more dots ('.')
            //println("${file_.getName().replaceAll(/.npy/, myConfig.out_name_append +'.h5')}")
        script:
            """
            /opt/anaconda3/envs/tompekin-basic/bin/python $projectDir/scripts/make_adorym_data.py --cfg $projectDir/conf/$params.confFile --Path2Unwarped $file_
            """
    }