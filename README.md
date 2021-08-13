[![Waffle.io - Columns and their card count](https://badge.waffle.io/dmbrady01/Monkey_frog.svg?columns=all)](https://waffle.io/dmbrady01/Monkey_frog) 


# Monkey_frog

Code for analyzing 2-photon imaging data with TTL event data

To setup your virtual environment, have `miniconda` installed for `python2.7x`, then run

```bash
$ conda env create -f environment.yml
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

If you want to run `process_data.py` from the command line, do the following:

```bash
$ python3 process_data.py
```

If you want to do it within python (or ipython), do:

```python
>>> exec(open("./process_data.py").read())
```