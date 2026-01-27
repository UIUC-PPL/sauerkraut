# Sauerkraut: Serializing Python's Control State
As `pickle` serializes Python's data state, `sauerkraut` serializes Python's control state.

Concretely, this library equips Python with the ability to stop a running Python function, serialize it to the network or disk, and resume later on. This is useful in different contexts, such as dynamic load balancing and checkpoint/restart.
Sauerkraut is designed to be lightning fast for latency-sensitive HPC applications. Internally, `FlatBuffers` are created to store the serialized state.

This library is still experimental, and subject to change.

## Examples

### Using Greenlets (Recommended)
The following example shows how to serialize a function's state using greenlets, which provides a cleaner interface:
```python
import sauerkraut
import greenlet
import numpy as np

def fun1(c):
    g = 4
    a = np.array([1, 2, 3])
    # Switch back to parent, preserving our state
    greenlet.getcurrent().parent.switch()
    
    # When we resume, continue here
    a += 1
    print(f'c={c}, g={g}, a={a}')
    return 3

# Create and start the greenlet
f1_gr = greenlet.greenlet(fun1)
f1_gr.switch(13)

# Serialize the greenlet's state
serframe = sauerkraut.copy_frame_from_greenlet(f1_gr, serialize=True)

# Save to disk
with open('serialized_frame.bin', 'wb') as f:
    f.write(serframe)

# Read from disk and resume execution
with open('serialized_frame.bin', 'rb') as f:
    read_frame = f.read()
code = sauerkraut.deserialize_frame(read_frame)
gr = greenlet.greenlet(sauerkraut.run_frame)
retval = gr.switch(code)
print(f"Done on the parent, child returned {retval}")
```

### Low-level API
This example demonstrates the lower-level frame copying and serialization API:
```python
import sauerkraut as skt
calls = 0

def fun1(c):
    global calls
    calls += 1
    g = 4
    for i in range(3):
      if i == 0:
          print("Copying frame")
          # copy_frame makes a copy of the current control + data states.
          # When we later run the frame (run_frame), execution will continue
          # directly after the call.
          # Therefore, this line will return twice:
          # The first time when the frame is copied,
          # the second time when the frame is resumed.
          frm_copy = skt.copy_current_frame()
      # Because copy_current_frame will return twice, we need to differentiate
      # between the different returns to avoid getting stuck in a loop!
      if calls == 1:
          # The copied frame does not see this write to g
          g = 5
          calls += 1
          # This variable is not serialized,
          # as it's not live
          hidden_inner = 55
          return frm_copy
      else:
          # When running the copied frame, we take this branch.
          # This line will run 3 times
          print(f'calls={calls}, c={c}, g={g}')

    calls = 0
    return 3

# Create and serialize a frame
frm = fun1(13)
serframe = skt.serialize_frame(frm)

# Save to disk
with open('serialized_frame.bin', 'wb') as f:
    f.write(serframe)

# Read from disk and resume execution
with open('serialized_frame.bin', 'rb') as f:
    read_frame = f.read()
code = skt.deserialize_frame(read_frame)
skt.run_frame(code)
```

## Installation

### From PyPI (Recommended)
```bash
pip install sauerkraut
```

### From Source
```bash
python3 -m pip install -r requirements.txt
python3 -m pip install .
```

### Using Docker
```bash
# Build the Docker image (takes 5-10 minutes)
docker build -t sauerkraut -f Dockerfile .

# Run the container
docker run --name="sauerkraut_img" -it library/sauerkraut

# Inside the container, you can run examples:
cd /sauerkraut/examples
python3 copy_then_serialize.py
```

## Compatibility
Sauerkraut leverages intimate knowledge of CPython internals, and as such is vulnerable to changes in the CPython API and VM.
Currently, Sauerkraut supports Python 3.13 and 3.14.
