# Optimizing semantic pointer representations for symbol-like processing in spiking neural networks

This repository contains the code to reproduce the results presented in the
paper "Optimizing semantic pointer representations for symbol-like processing in
spiking neural networks".

## Requirements

* [Python 2.7](https://www.python.org/) (Python 3 might work with minor changes,
  but has not been tested.)

The following Python packages that should be `pip` installable:

* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [pandas](http://pandas.pydata.org/)
* [nengo_spinnaker](https://github.com/project-rig/nengo_spinnaker)
* [Jupyter](https://jupyter.org/)
* [matplotlib](http://matplotlib.org/)
* [Seaborn](http://stanford.edu/~mwaskom/software/seaborn/)

The following Python packages that are not yet `pip` installable in the required
version:

* [Nengo](https://github.com/nengo/nengo) at commit 17f5e81bd432aaa272b3d69ec2ba179f6f0bf6ef (or newer)
* [psyrun](https://github.com/jgosmann/psyrun) at commit 852da71434873a060d88fbc00a7bc5df6bb9e6b6

To install these to python packages run the following commands:

```shell
git clone https://github.com/nengo/nengo.git
git clone https://github.com/jgosmann/psyrun.git
cd nengo
git checkout 17f5e81bd432aaa272b3d69ec2ba179f6f0bf6ef
python setup.py install --user
cd ../psyrun
git checkout 852da71434873a060d88fbc00a7bc5df6bb9e6b6
python setup.py install --user
```

The code in this repository will likely work with the upcoming Nengo 2.1 release
as well. Thus, following the release installing this specific Nengo version
should not be required anymore and a `pip install nengo` should suffice. The
`psyrun` API might still change at this point and I recommend using the version
given here.

## Usage

To generate all the data change to the repositories root directory and run
`psy-doit` twice. This will require a configured SpiNNaker board connected to
the computer. To only generate specific data sets run `psy-doit <task_name>`. To
get a list of all task names run `psy-doit list`.

The tasks ending in `_less_neurons` require data from the task with the same
name without the suffix. As long as that data does not exist, the
`_less_neurons` tasks will do nothing. (That is why `psy-doit` has to be run
twice to generate all the data.)

To generate the plots from the data open `Data analysis.ipynb` with Jupyter
notebook and run the cells.
