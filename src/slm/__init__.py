"""
slm: library code of the analysis pipeline (see README.md).

The numbered scripts in scripts/ import these modules after scripts/_setup.py has put src/ on
sys.path.  Modules, in pipeline order:

    params         analysis parameters and file locations (imported as P by every script)
    io             JSON helpers and a timer for the logs
    corpus         streaming of the ConvoKit corpora (steps 1 and 2)
    cohort         posting rate, activity criteria and onsets (steps 2 and 3)
    windows        two-week windows and the text shown to the model (steps 3 and 4)
    prompts        symptom battery and prompt construction
    parse          parsing of model responses and window x symptom tables (step 5)
    nmf            nonnegative matrix factorization and staNMF stability
    psychometrics  weighted alpha, item discrimination, test-retest reliability
    var            panel vector autoregression
    states         UMAP embedding and Dirichlet process Gaussian mixture states
    markov         mixtures of Markov chains (dynamical regimes)
    lexicon        VADER, Empath and style features (step 13)
    plotting       figure style; pptx_export: editable PowerPoint versions of the figures
"""
