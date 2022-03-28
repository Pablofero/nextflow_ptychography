import nextflow.cli.Launcher
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.Files
def copy_config_to_output_folder(String folder){
    cmd_with_nextflow = session.commandLine.split() // get the cmd line string, aka. smth like "nextflow run main.nf -resume -params-file config.yaml", and split it
    String[] cmd = Arrays.copyOfRange(cmd_with_nextflow, 1, cmd_with_nextflow.length) // get rid of the "nextflow" at the beginning
    launcher  = new Launcher().parseMainArgs(cmd) // parse it identically to nextflow
    paramsFileStr = launcher.command.paramsFile

    Path paramsFile = Paths.get(paramsFileStr); 
    Path copyparamsFile = Paths.get(folder, paramsFileStr);
    copyparamsFile.rollFile() //don't overwrite, rename existing files! The newest will have no number at the end
    Files.copy(paramsFile, copyparamsFile);
}
