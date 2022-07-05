import pyopencl as cl
import numpy as np

cl_ctx, cl_queue = None, None
devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
if len(devices) == 0:
    devices = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)
cl_ctx = cl.Context(devices=devices)
cl_queue = cl.CommandQueue(cl_ctx)

@profile
def test():
    for i in range(1000):
        shape = (512, 512, 4)
        size = int(np.prod(shape)) * 4
        # create buffer
        flags = cl.mem_flags.READ_WRITE
        buffer = cl.Buffer(cl_ctx, flags, size, hostbuf=None)
        # fill buffer
        cl.enqueue_fill_buffer(cl_queue, buffer, np.float32(i), 0, size)

test()
