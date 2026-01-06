import sauerkraut
import numpy as np

calls = 0


def fun1(c):
    global calls
    calls += 1
    g = 4
    local_array = np.zeros((5, 5))
    local_array += 1
    frame_bytes = sauerkraut.copy_current_frame(serialize=True)
    if calls == 1:
        g = 5
        local_array += 1
        calls += 1
        hidden_inner = 55
        return frame_bytes
    else:
        print(f"local_array=\n{local_array}")
        assert np.all(local_array == 1)

    calls = 0
    return 3


frame_bytes = fun1(13)
with open("serialized_frame.bin", "wb") as f:
    f.write(frame_bytes)
with open("serialized_frame.bin", "rb") as f:
    read_frame = f.read()
retval = sauerkraut.deserialize_frame(read_frame, run=True)

print("Function returned with:", retval)
