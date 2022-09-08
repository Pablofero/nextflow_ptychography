import nextflow.dag.DAG
import nextflow.dag.DagRenderer
import nextflow.dag.GraphvizRenderer
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import nextflow.Session
import nextflow.cli.Launcher

def render_dag(String folder,String format = 'png'){
    cmd_with_nextflow = session.commandLine.split() // get the cmd line string, aka. smth like "nextflow run main.nf -resume -params-file config.yaml", and split it
    String[] cmd = Arrays.copyOfRange(cmd_with_nextflow, 1, cmd_with_nextflow.length) // get rid of the "nextflow" at the beginning
    launcher  = new Launcher().parseMainArgs(cmd) // parse it identically to nextflow
    paramsFile = launcher.command.paramsFile - ~/\.\w+$/ // get rid ot th extension(.yaml)

    // print(this.binding.variables.each {k,v -> println "${v.dump()} $k = $v"})//print all the variables
    // print(([:] as CmdRun).getFields())

    DAG dag = session.getDag() // get the execution graph
    dag.normalize() // get it ready to render
    Path file = Paths.get(folder, paramsFile+'.'+format); 
    file.rollFile() //don't overwrite, rename existing files! The newest will have no number at the end
    DagRenderer renderer = new GraphvizRenderer(file.baseName, format)
    renderer.renderDocument(dag, file)
}