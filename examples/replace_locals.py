import sauerkraut as skt
import greenlet


def fun1():
    a = 1
    b = 2
    greenlet.getcurrent().parent.switch()
    return a + b


def main():
    gr1 = greenlet.greenlet(fun1)
    gr1.switch()
    print("Back in main")
    serframe = skt.copy_frame_from_greenlet(gr1, serialize=True)
    res = skt.deserialize_frame(serframe, replace_locals={"a": 5}, run=True)
    print(f"The result is {res}")

    deserframe = skt.deserialize_frame(serframe)
    res = skt.run_frame(deserframe, replace_locals={"a": 9})
    print(f"The result is {res}")

    deserframe = skt.deserialize_frame(serframe)
    res = skt.run_frame(deserframe, replace_locals={"b": 35})
    print(f"The result is {res}")


if __name__ == "__main__":
    main()
