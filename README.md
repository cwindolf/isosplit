# isosplit

a numpy implementation of J. Magland and A. Barnett, [Unimodal clustering using isotonic regression: ISO-SPLIT](https://arxiv.org/abs/1508.04841).

there is a much (about 10x) faster Python wrapper of the C++ implementation by those authors [here](https://github.com/magland/isosplit5_python), which should be preferred over this implementation in all cases. for Matlab, see the [official implementation](https://github.com/flatironinstitute/isosplit5).

this repository is unofficial and in no way endorsed by those authors. this code is (or, claims to be) a direct translation to Python of the Matlab implementation.

### demo

run `pip install -e .` in this directory, and then check out the demo notebook in `demo/`.
