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

### From the command line
If you want to run `process_data.py` from the command line, do the following:

```bash
$ python3 process_data.py
```
By default, it will read parameters from `params.json`, but you can make your own file and read it in with the `-f` flag.

```bash
$ python3 process_data.py -f my_parameters.json
```

### From ipython or spyder
```python
[1]: !python3 process_data.py -f params.json
```
You can also use the python shell commands below

### From the python shell
With no separate parameter file:
```python
>>> exec(open("./process_data.py").read())
```

If you want to examine the output
```python
>>> from process_data import process_data
>>> trials, segment_list = process_data('my_parameters.json')
```

If you do have your own parameter file, you can us the `os` or `subprocess` modules:
```python
>>> import os
>>> os.system("python3 process_data.py -f my_parameters.json")
```

```python
>>> import subprocess
>>> subprocess.Popen(["python3", "process_data.py", "-f", "params.json"])
```
