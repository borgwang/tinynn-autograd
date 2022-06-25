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
  - handle pyopencl array broadcasting
  - sum along specify asix
  - [x] contiguous array is still contiguous after transpose
  - [] implement movement ops
  - [] buffer instead of array
- mlops
  - layers/loss/optimizer
- misc
  - create gpu tensor directly

### License

MIT

