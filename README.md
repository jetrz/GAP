# GAP
GAP = Genome Assembly Postprocessor :-)

## How To Use
<table>
  <tr>
    <th style="width: 75px;">Function</th>
    <th style="width: 75px;">Info</th>
    <th style="width: 250px;">Input Files</th>
    <th style="width: 250px;">Output Files</th>
  </tr>
  <tr>
    <td>Preprocessing</td>
    <td>Preprocesses the GFA, PAF and Error-Corrected Reads FASTA into the relevant files needed for the other steps. <br> If it is a Hifiasm run, the GFA path should be a .p_ctg.gfa file. If it is a GNNome run, the GFA file should be a .bp.raw.r_utg.gfa file.</td>
    <td>- GFA file <br>- PAF file <br>- EC Reads FASTA</td>
    <td>- Graph <br>- Node-to-sequence (n2s) <br>- Read-to-node (r2n) <br>- Read-to-sequence (r2s) <br>- Processed PAF <br></td>
  </tr>
  <tr>
    <td>Baseline Generation</td>
    <td>Generates the baseline assembly in the required format.</td>
    <td>
    <b>For GNNome</b>
    - Graph <br>- Node-to-sequence (n2s) <br>- Genome Reference<br>
    <b>For Hifiasm</b>
    - GFA file <br>- Read-to-sequence (r2s) <br>- Genome Reference<br>
    </td>
    <td>- Baseline walks & assembly</td>
  </tr>
  <tr>
    <td>Postprocessing</td>
    <td>Runs postprocessing step to improve assembly.</td>
    <td>- Contigs <br>- Graph <br>- Node-to-sequence (n2s) <br>- Read-to-node (r2n) <br>- Read-to-sequence (r2s) <br>- Processed PAF <br>- Genome Reference</td>
    <td>- Final assembly</td>
  </tr>
</table>

## Directory
The codebase is split into the three main functions, each with their respective directory.

    - main.py  		              Main script to run.
    - config.yaml               Configs to be set. Ensure that the genome you are running has its info in the specified format.
    - preprocess/             
        - preprocess.py         Main script to run the various pre-processing steps.
        - gfa_util.py           Script to pre-process GFA file.
        - fasta_util.py         Script to pre-process error-corrected reads FASTA file.
        - paf_util.py           Script to pre-process PAF file.
    - generate_baseline/
        - generate_baseline.py  Main script to generate the baseline walks and assembly.
        - hifiasm_decoding.py   Generates baseline from Hifiasm's final GFA.
        - gnnome_decoding.py    Generates baseline from GNNome's graph. Basic version of GNNome's decoding step.
        - SymGatedGCN.py        SymGatedGCN layer from GNNome.
    - postprocess/
        - postprocess.py        Script for postprocessing pipeline.
