## tinynn-autograd

tinynn-autograd is a deep learning framework with automatic differentiation, built upon [tinynn](https://github.com/borgwang/tinynn).


### install

```bash
git clone https://github.com/borgwang/tinynn-autograd
cd tinynn-autograd
pip3 install -r requirements.txt
```

### run example

```bash
cd tinynn-autograd
python3 examples/mnist/run.py
```

### Resources

- blog post (in Chinese): [Automatic Differentiation Tutorial](https://borgwang.github.io/dl/2019/09/15/autograd.html)
- [tinynn](https://github.com/borgwang/tinynn)


### TODOs

- llops
  - reduce ops
    - [x] reduce along axis
    - [x] reduce support 4D array
    - [x] reduction when size is not power of 2
  - process ops
    - conv op
  - movement ops
    - [x] reshape op
    - [x] expand op
    - [x] contiguous op
    - slice op
  - binary ops
  - unary ops
- mlops
  - conv/transpose_conv
  - refactor layers/loss/optimizer
- misc
  - [x] create gpu tensor directly
  - [x] initializer
  - unify gpu_ops and ops
- abbstraction
  - [x] buffer instead of array
  - wrap numpy array to CPUArray
- unit testing
- speedup
  - benchmark
    - cpu w/o autugrad: 0.48s per epoch (intel i7 6 core, 2.6GHz)
    - cpu w/ autograd: 2.2s per epoch
    - gpu w/ autograd: 2.4s per epoch (AMD)
  - graph optimization
    - [x] avoid repeated backward (2x faster)
    - [x] avoid uneccessary buffer creatation (20% faster)
    - [x] don't wait (2x faster)
    - [x] avoid uneccessary gradient add (30% faster)
    - combine nodes
  - gpuops optimization
    - [x] drelu
  - cache cl buffer then we don't have to recreate each time

### License

MIT

