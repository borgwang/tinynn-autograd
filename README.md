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
    - [x] matmul supprot 4d array
    - conv op
  - movement ops
    - [x] reshape op
    - [x] expand op
    - [x] contiguous op
    - slice op
  - binary ops
    - support 4d array
  - unary ops
    - support 4d array
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
- graph optimization
  - avoid repeated backward
  - combine nodes
- unit testing

### Design

### License

MIT

