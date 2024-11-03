# GAP
GAP = Genome Assembly Postprocessor :-)

The codebase is split into three main functions, each with their respective directory.

Directory:
    main.py:                Main script to run.
    config.yaml             Configs to be set. Ensure that the genome you are running has its info in the specified format.
    preprocess/             
        preprocess.py       Main script to run the various pre-processing steps.
        gfa_util.py         Script to pre-process GFA file.
        fasta_util.py       Script to pre-process reads FASTA file.
        paf_util.py         Script to pre-process PAF file.
    gnnome_decoding/
        gnnome_decoding.py  Basic version of GNNome's decoding step.
        SymGatedGCN.py      SymGatedGCN layer from GNNome.
    postprocess/
        postprocess.py      Script for postprocessing pipeline.
