# SuPar Sparse Coordinate Check Reproduction

This folder contains code to reproduce Figure 5 from [Sparse maximal update parameterization: A holistic approach to sparse training dynamics](https://arxiv.org/abs/2405.15743). To reproduce this figure yourself, first download the tiny shakespeare dataset by running `python data/shakespeare_char/prepare.py`.

Then to collect the results for the sparse coordinate check run:
```
bash supar_examples/coord_check_shakespeare_char/supar/run.sh
bash supar_examples/coord_check_shakespeare_char/mup/run.sh
bash supar_examples/coord_check_shakespeare_char/sp/run.sh
```

Finally `supar_examples/coord_check_shakespeare_char/plot.ipynb` contains the code to produce the following figure:

TODO
