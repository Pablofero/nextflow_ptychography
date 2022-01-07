## Nextflow Preprocessing
___
### To run:
```
nextflow main.nf -resume -name run1
```
The resume option uses cached results (to mark as dirty, modify the corresponding .nf).
The ```main.nf``` workflow saves it's outputs to a folder called ```output``` by default.

Nextflow [Documentation](https://www.nextflow.io/docs/latest/) and [Patterns](https://nextflow-io.github.io/patterns/index.html) (not in DSL 2, but still valuable)
___
### Structure
```
├─.nextflow/ - (autogenerated) nextflow internals
├─ conf/ - .json with the configurations of the scripts. These files are the tunable parameters
├─ data/ - input files
├─ nextflow_modules/ - do not need to change
├─ output/ - (autogen.)
├─ scripts/ - scripts - the code that is called
├─ work/ - (autogen.) nextflow's cache, you can look here to get valuable information on how your script run (the 'random' characters)
├─ main.nf - main nextflow workflow
├─ nextflow.config - nextflow execution configurations (how many CPUs/RAM, SLURM, etc.)
```
___
Python scripts written by Tom Pekin, workflow, parallelisation and modifications of scripts by Pablo Fernández Robledo. Also Wouter Van den Broek with the unwarping code for the Dectris detector

Contact: [tcpekin@gmail.com](mailto:tcpekin@gmail.com) and [robledop@physik.hu-berlin.de](mailto:robledop@physik.hu-berlin.de) or [pablofrldp@gmail.com](mailto:pablofrldp@gmail.com)
