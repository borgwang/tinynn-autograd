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
  - [x] handle pyopencl array broadcasting
  - [x] reduce along axis
  - [x] contiguous array is still contiguous after transpose
  - [x] implement movement ops
  - [x] buffer instead of array
  - [x] reduce support 4D array
  - [x] matmul supprot 4d array
  - [x] noncontiguous array
- mlops
  - layers/loss/optimizer
- misc
  - [x] create gpu tensor directly
  - [] initializer
  - [] unify gpu_ops and ops
- abbstraction
  - wrap numpy array to CPUArray

### Design

### License

MIT

