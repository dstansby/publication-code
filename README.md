This repository contains all the code needed to reproduce figures contained
within all of my first author publications. Each publication has it's
own folder; see [index.csv](https://github.com/dstansby/publication-code/blob/master/index.csv)
for which folder corresponds to each publication.

Acknowledging
-------------
You are free to use, re-use, and re-mix any of the code found here for your own uses. If any of
the code contributes to a publication I would appreciate it if you cited the corresponding paper
that I originally wrote the code for, with a acknowledgement similar to
> "This paper made use of software originally developed for Stansby et al. (year)".

Running
-------

To run the code you will need an
installation of [conda](https://conda.io/). When this is done change to a
specific publication directory, and an environment can be set up to run the code
using the following shell commands:

```bash
conda env create -f environment.yml --name pub-env
source activate pub-env
```

All of the code is stored in jupyter notebooks. To run a notebook, just run

```bash
jupyter notebook
```

You can then select and run files individually.
