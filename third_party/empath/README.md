`categories.tsv` is the 194-category lexicon shipped with the `empath` 0.89
client (PyPI; Fast, Chen & Bernstein 2016, "Empath: Understanding Topic
Signals in Large-Scale Text", CHI '16; github.com/Ejhfast/empath-client).
The PyPI package no longer builds on Python 3.11, so the lexicon is vendored
here and applied by `src/slm/lexicon.py` (lower-cased, punctuation-stripped
whitespace tokens; counts per token, as empath's `analyze(normalize=True)`).
