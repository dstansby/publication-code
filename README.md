This repository contains all the code needed to reproduce figures contained
within all of my first author publications. Each publication has it's
own folder; see index.csv for which folder corresponds to each publication.

All code is run on python 3.6. To run the code you will need an installation
of [conda](https://conda.io/). When this is done, an environment can be set up
to run the code using the following shell commands:

```bash
conda create --name pub-env python=3.6
source activate pub-env
pip install -r requirements.txt
```

All of the code is stored in jupyter notebooks. To run a notebook, just run

```bash
jupyter notebook
```

You can then select and run files individually.
