# Monkey_frog

Code for analyzing 2-photon imaging data with TTL event data

## Setting up your virtual environment

To setup your virtual environment, have `miniconda` installed for `python3.8x`, then run

```bash
$ conda env create -f environment.yaml
```

Before running any code, make sure to enter your virtual environment

```bash
$ conda activate monkey_frog
```

When finished for the day

```bash
$ conda deactivate
```
will get you out of your virtual environment
## Running `process_data.py`

If you want to run `process_data.py` from the command line, do the following:

```bash
$ python3 process_data.py
```
By default, it will read parameters from `params.json`, but you can make your own file and read it in with the `-f` flag.

```bash
$ python3 process_data.py -f my_parameters.json
```

If you want to do it within python (or ipython) and do not have any additional arguments, the easiest is to do:

```python
>>> exec(open("./process_data.py").read())
```

If you do have your own parameter file, you can us the `os` or `subprocess` modules:

```python
>>> import os
>>> os.system("python3 process_data.py -f my_parameters.json")
```
