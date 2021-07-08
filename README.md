# isosplit

a numpy implementation of J. Magland and A. Barnett, [Unimodal clustering using isotonic regression: ISO-SPLIT](https://arxiv.org/abs/1508.04841).

there is a much (about 10x) faster Python wrapper of the C++ implementation by those authors [here](https://github.com/magland/isosplit5_python), which should be preferred over this implementation in all cases. for Matlab, see the [official implementation](https://github.com/flatironinstitute/isosplit5).

this repository is unofficial and in no way endorsed by those authors. this code is (or, claims to be) a direct translation to Python of the Matlab implementation.


### demo

run `pip install -e .` in this directory, and then check out the demo notebook in `demo/`.


### up-down isotonic regression

this module also exports functions `up_down_isotonic_regression` and `down_up_isotonic_regression`, which are translations of the original authors' least squares up-down isotonic regression subroutine, which adapts the Pool-Adjacent-Violators algorithm from the usual isotonic regression setting


### isosplit1d

this is my own personal riff on what can be done with the `isocut` primitive to solve the clustering problem for one-dimensional data. the idea is to fix a threshold for the dip score statistic and recursively split the data until there are no remaining significant cuts. optional parameters for minimum cluster size and minimum spatial extent of clusters are also included, and the discovered cut points which divide the spatial domain are returned so that they can be used for extrapolation.
