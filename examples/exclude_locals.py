import sauerkraut as skt
import greenlet


def fun1():
    a = 1
    b = 2
    c = 3
    greenlet.getcurrent().parent.switch()
    d = 0
    return a + b + c + d


def main():
    gr1 = greenlet.greenlet(fun1)
    gr1.switch()
    serframe = skt.copy_frame_from_greenlet(
        gr1, exclude_locals={"a", "c"}, serialize=True
    )
    res = skt.deserialize_frame(serframe, run=True, replace_locals={"a": 22, "c": 35})
    print(f"The result is {res}")

    deserframe = skt.deserialize_frame(serframe)
    res = skt.run_frame(deserframe, replace_locals={"a": 4, "c": 5})
    print(f"The result is {res}")

    deserframe = skt.deserialize_frame(serframe)
    try:
        res = skt.run_frame(deserframe, replace_locals={"c": 200})
    except TypeError:
        print(
            "When you forget to replace an excluded local, 'None' is used in its place!"
        )


if __name__ == "__main__":
    main()
