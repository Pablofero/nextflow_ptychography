#!/bin/sh
nextflow run -name output_hbn_tile_0_overlap_0_again main.nf -params-file config_tom_hbn_copy_0.yaml --output_folder output_hbn_tile_0_overlap_again
nextflow run -name output_hbn_tile_1_overlap main.nf -params-file config_tom_hbn_copy.yaml --output_folder output_hbn_tile_1_overlap || true
nextflow run -name output_hbn_tile_2_overlap_2 main.nf -params-file config_tom_hbn_copy_2.yaml --output_folder output_hbn_tile_2_overlap || true
nextflow run -name output_hbn_tile_3_overlap_2 main.nf -params-file config_tom_hbn_copy_3.yaml --output_folder output_hbn_tile_3_overlap || true



nextflow log -f  duration,status,workdir,name

output_hbn_tile_3_overlap_again