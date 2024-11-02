# GAP
GAP = Genome Assembly Postprocessor :-)

Directory:
    main.py: 
    config.yaml             Configs to be set. Ensure that the genome you are running has its info in the specified format.
    preprocess/             Scripts to pre-process data files.
        preprocess.py       Main script to run the various pre-processing steps
        gfa_util.py         Script to pre-process GFA file.
        fasta_util.py       Script to pre-process reads FASTA file.
        paf_util.py         Script to pre-process PAF file.
    postprocess/
    misc/                   Miscellaneous files
        utils.py            Minor utility functions.