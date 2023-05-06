# DEMAP - a python DEM Analysis Package for geomorphologist

Author: Jingtao Lai (lai@gfz-potsdam.de)

`DEMAP` is a set of python-based topographic analysis tools that help
geomorphologists decode information from digital maps.

## Installation

Download the source code and install it using `pip` locally:

```Shell
git clone https://github.com/laijingtao/demap.git
pip install ./demap
```

## Accelerate with `Numba`

Many methods in `DEMAP` can be accelerated using `Numba`. If `Numba` is already
installed, these methods will automatically detect it and run much faster.
[Here](https://numba.pydata.org/numba-doc/latest/user/installing.html) is the
guide for installing `Numba`.
