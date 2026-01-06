import sauerkraut
import greenlet
import numpy as np
import inspect


def SerializeIt():
    print("Copying the frame")
    serframe = sauerkraut.copy_frame(inspect.currentframe(), serialize=True)
    print("Switching to the parent with the frame")
    greenlet.getcurrent().parent.switch(serframe)


def fun1(c):
    g = 4
    a = np.array([1, 2, 3])
    for i in range(10):
        print(f"Iteration {i}")
        if i == 5:
            SerializeIt()
        a += 1
        print(f"c={c}, g={g}, a={a}")
    print("The loop is done")
    return a


f1_gr = greenlet.greenlet(fun1)
serframe = f1_gr.switch(13)
# serframe = sauerkraut.copy_frame_from_greenlet(f1_gr, serialize=True)
print("On the parent, about to serialize it")
with open("serialized_frame.bin", "wb") as f:
    f.write(serframe)
with open("serialized_frame.bin", "rb") as f:
    read_frame = f.read()
for i in range(10):
    print(f"Iteration {i}")
    code = sauerkraut.deserialize_frame(read_frame)
    print("Deserialized the frame")
    gr = lambda: sauerkraut.run_frame(code)
    gr = greenlet.greenlet(gr)
    result = gr.switch()
    print(f"The result is {result}")
print("Done")
