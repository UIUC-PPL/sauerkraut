import sauerkraut

calls = 0


def fun1(c):
    global calls
    calls += 1
    g = 4
    print("About to start the loop")

    for i in range(5):
        continue
        fun2()
    for i in range(3):
        for j in range(6):
            if i == 0 and j == 0:
                print("Copying frame")
                frm_copy = sauerkraut.copy_current_frame()
                #
            if calls == 1:
                g = 5
                calls += 1
                hidden_inner = 55
                return frm_copy
            else:
                print(f"calls={calls}, c={c}, g={g}")

    calls = 0
    return 3


frm = fun1(13)
serframe = sauerkraut.serialize_frame(frm)
with open("serialized_frame.bin", "wb") as f:
    f.write(serframe)
with open("serialized_frame.bin", "rb") as f:
    read_frame = f.read()
code = sauerkraut.deserialize_frame(read_frame)
sauerkraut.run_frame(code)

print("Done")
