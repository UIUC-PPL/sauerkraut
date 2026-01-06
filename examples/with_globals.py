import sauerkraut as skt
import greenlet
import numpy as np


def fun1():
    a = np.array([1, 2, 3])
    a += 1
    greenlet.getcurrent().parent.switch()
    a += 1
    print(f"Mean of a is {np.mean(a)}")
    return a


def main():
    gr = greenlet.greenlet(fun1)
    gr.switch()
    serialized = skt.copy_frame_from_greenlet(gr, serialize=True)
    with open("serialized_globals_example.bin", "wb") as f:
        f.write(serialized)


if __name__ == "__main__":
    main()
