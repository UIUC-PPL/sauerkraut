import sauerkraut

calls = 0

"""
In this example, we show an optimization where we skip copies.
In the `copy_frame` call, we set serialize to True. This will serialize the frame
and return the bytes, rather than returning a copy of the frame that must be copied later.
This is important if our locals are large (e.g., numpy arrays).

The call to deserialize_frame is optimized as well with the 'run' flag set to True.
In this case, the new frame will be directly allocated on Python's frame stack where it can
be run directly. It is allocated on the frame stack and immediately run, so the returned value is that
of the wrapped function.
If run is set to False (the default), the frame will be allocated on the regular heap.
Before it can be run, it must be copied to the frame stack. This is done by calling sauerkraut.run_frame.
"""


def fun1(c):
    global calls
    calls += 1
    g = 4
    frame_bytes = sauerkraut.copy_current_frame(serialize=True)
    if calls == 1:
        g = 5
        calls += 1
        hidden_inner = 55
        return frame_bytes
    else:
        print(f"calls={calls}, c={c}, g={g}")

    calls = 0
    return 3


frame_bytes = fun1(13)
with open("serialized_frame.bin", "wb") as f:
    f.write(frame_bytes)
with open("serialized_frame.bin", "rb") as f:
    read_frame = f.read()
retval = sauerkraut.deserialize_frame(read_frame, run=True)

print("Function returned with:", retval)
