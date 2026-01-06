import sauerkraut as skt

import greenlet


def fun1():
    a = 1
    b = 2
    c = 3
    for i in range(25):
        if i > 10:
            a += 3
        if i > 2:
            b += 4
        if i == 24:
            # switch only on the final iteration
            greenlet.getcurrent().parent.switch()
        if b > 28:
            print(f"b is {b}")
    d = 0
    return a + b + d


def main():
    gr1 = greenlet.greenlet(fun1)
    gr1.switch()
    serframe = skt.copy_frame_from_greenlet(
        gr1, exclude_dead_locals=True, serialize=True
    )
    res = skt.deserialize_frame(serframe, run=True, replace_locals={"a": 22, "c": 35})
    print(f"The result is {res}")

    # deserframe = skt.deserialize_frame(serframe)
    # res = skt.run_frame(deserframe, replace_locals={'a': 4, 'c': 5})
    # print(f"The result is {res}")

    # deserframe = skt.deserialize_frame(serframe)
    # try:
    #     res = skt.run_frame(deserframe, replace_locals={'c': 200})
    # except TypeError as e:
    #     print("When you forget to replace an excluded local, 'None' is used in its place!")


if __name__ == "__main__":
    main()
