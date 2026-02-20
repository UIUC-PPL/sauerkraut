import sauerkraut as skt
from sauerkraut import liveness
import greenlet
import numpy as np
import importlib
import os
import sys
import tempfile
import textwrap
import uuid

calls = 0


def test1_fn(c):
    global calls
    calls += 1
    g = 4
    frm_copy = skt.copy_current_frame(sizehint=128)
    if calls == 1:
        g = 5
        calls += 1
        hidden_inner = 55
        return frm_copy
    else:
        print(f"calls={calls}, c={c}, g={g}")

    calls = 0
    return 3


def test_copy_then_serialize():
    global calls
    calls = 0
    frm = test1_fn(55)
    serframe = skt.serialize_frame(frm, sizehint=5)
    with open("serialized_frame.bin", "wb") as f:
        f.write(serframe)
    with open("serialized_frame.bin", "rb") as f:
        read_frame = f.read()
    code = skt.deserialize_frame(read_frame)
    retval = skt.run_frame(code)
    assert retval == 3
    print("Test 'copy_then_serialize' passed")


def test2_fn(c):
    global calls
    calls += 1
    g = 4
    frame_bytes = skt.copy_current_frame(serialize=True)
    if calls == 1:
        g = 5
        calls += 1
        hidden_inner = 55
        return frame_bytes
    else:
        print(f"calls={calls}, c={c}, g={g}")
        retval = calls + c + g
    return retval


def test_combined_copy_serialize():
    global calls
    calls = 0

    frame_bytes = test2_fn(13)
    with open("serialized_frame.bin", "wb") as f:
        f.write(frame_bytes)
    with open("serialized_frame.bin", "rb") as f:
        read_frame = f.read()
    retval = skt.deserialize_frame(read_frame, run=True)

    print("Function returned with:", retval)
    assert retval == 19
    print("Test combined_copy_serialize passed")


def for_loop_fn(c):
    global calls
    calls += 1
    g = 4
    print("About to start the loop")

    sum = 0
    for i in range(3):
        for j in range(6):
            sum += 1
            if i == 0 and j == 0:
                print("Copying frame")
                frm_copy = skt.copy_current_frame(serialize=True)
                #
            if calls == 1:
                g = 5
                calls += 1
                hidden_inner = 55
                return frm_copy
            else:
                print(f"calls={calls}, c={c}, g={g}")

    calls = 0
    return sum


def test_for_loop():
    global calls
    calls = 0

    serframe = for_loop_fn(42)
    with open("serialized_frame.bin", "wb") as f:
        f.write(serframe)
    with open("serialized_frame.bin", "rb") as f:
        read_frame = f.read()
    code = skt.deserialize_frame(read_frame)
    iters_run = skt.run_frame(code)

    assert iters_run == 18
    print("Test 'for_loop' passed")


def greenlet_fn(c):
    a = np.array([1, 2, 3])
    greenlet.getcurrent().parent.switch()
    a += 1
    print(f"c={c}, a={a}")
    greenlet.getcurrent().parent.switch()
    return 3


def test_greenlet():
    gr = greenlet.greenlet(greenlet_fn)
    gr.switch(13)
    serframe = skt.copy_frame_from_greenlet(gr, serialize=True)
    with open("serialized_frame.bin", "wb") as f:
        f.write(serframe)
    with open("serialized_frame.bin", "rb") as f:
        read_frame = f.read()
    code = skt.deserialize_frame(read_frame)
    gr = greenlet.greenlet(skt.run_frame)
    gr.switch(code)
    print("Test 'greenlet' passed")


def replace_locals_fn(c):
    a = 1
    b = 2
    greenlet.getcurrent().parent.switch()
    return a + b + c


def test_replace_locals():
    gr = greenlet.greenlet(replace_locals_fn)
    gr.switch(13)
    serframe = skt.copy_frame_from_greenlet(gr, serialize=True)
    code = skt.deserialize_frame(serframe)
    res = skt.run_frame(code, replace_locals={"a": 9})
    print(f"The result is {res}")
    assert res == 24

    serframe = skt.copy_frame_from_greenlet(gr, serialize=True)
    code = skt.deserialize_frame(serframe)
    res = skt.run_frame(code, replace_locals={"b": 35})
    print(f"The result is {res}")
    assert res == 49

    serframe = skt.copy_frame_from_greenlet(gr, serialize=True)
    code = skt.deserialize_frame(serframe)
    res = skt.run_frame(code, replace_locals={"a": 9, "b": 35, "c": 100})
    print(f"The result is {res}")
    assert res == 144

    print("Test 'replace_locals' passed")


def exclude_locals_greenletfn(c):
    a = 1
    b = 2
    greenlet.getcurrent().parent.switch()
    return a + b + c


def exclude_locals_current_frame_fn(c, exclude_locals=None):
    global calls
    calls += 1
    g = 4
    frame_bytes = skt.copy_current_frame(
        serialize=True, exclude_locals=exclude_locals, exclude_dead_locals=False
    )
    if calls == 1:
        g = 5
        calls += 1
        hidden_inner = 55
        return frame_bytes
    else:
        print(f"calls={calls}, c={c}, g={g}")
        retval = calls + c + g
    return retval


def test_exclude_locals_greenlet():
    gr = greenlet.greenlet(exclude_locals_greenletfn)
    gr.switch(13)
    serframe = skt.copy_frame_from_greenlet(
        gr, serialize=True, exclude_locals={"a"}, sizehint=500, exclude_immutables=True
    )
    deserframe = skt.deserialize_frame(serframe)
    try:
        res = skt.run_frame(deserframe)
    except TypeError:
        print(
            "When you forget to replace an excluded local, 'None' is used in its place!"
        )

    result = skt.deserialize_frame(serframe, replace_locals={"a": 9}, run=True)
    assert result == 24

    gr2 = greenlet.greenlet(exclude_locals_greenletfn)
    gr2.switch(13)
    serframe = skt.copy_frame_from_greenlet(
        gr2, serialize=True, exclude_locals=["c", "b"], exclude_immutables=True
    )
    deserframe = skt.deserialize_frame(serframe)
    result = skt.run_frame(deserframe, replace_locals={"c": 100, "b": 35})
    assert result == 136
    print("Test 'exclude_locals_greenlet' passed")


def test_exclude_locals_current_frame():
    global calls
    calls = 0
    exclude_locals = {"exclude_locals", "g"}
    frm_bytes = exclude_locals_current_frame_fn(13, exclude_locals)
    result = skt.deserialize_frame(frm_bytes, run=True, replace_locals={"g": 8})
    print(f"The result is {result}")
    assert result == 23

    calls = 0
    exclude_locals = {"exclude_locals", "c"}
    frm_bytes = exclude_locals_current_frame_fn(13, exclude_locals)
    result = skt.deserialize_frame(frm_bytes, run=True, replace_locals={"c": 100})
    print(f"The result is {result}")
    assert result == 106

    calls = 0
    exclude_locals = {"exclude_locals", 0}
    frm_bytes = exclude_locals_current_frame_fn(13, exclude_locals)
    result = skt.deserialize_frame(frm_bytes, replace_locals={0: 25}, run=True)
    print(f"The result is {result}")
    assert result == 31

    print("Test 'exclude_locals_current_frame' passed")


def test_exclude_locals():
    test_exclude_locals_greenlet()
    test_exclude_locals_current_frame()


def _copy_frame_and_switch():
    import inspect

    frame_bytes = skt.copy_frame(inspect.currentframe(), serialize=True)
    greenlet.getcurrent().parent.switch(frame_bytes)


def copy_frame_target_fn(c):
    x = 100
    total = 0
    for i in range(3):
        total += i
        if i == 1:
            _copy_frame_and_switch()
    return x + c + total


def test_copy_frame():
    gr = greenlet.greenlet(copy_frame_target_fn)
    frame_bytes = gr.switch(50)
    code = skt.deserialize_frame(frame_bytes)
    gr2 = greenlet.greenlet(skt.run_frame)
    result = gr2.switch(code)
    assert result == 153
    print("Test 'copy_frame' passed")


def resume_greenlet_fn(c):
    a = 5
    greenlet.getcurrent().parent.switch()
    a += c
    return a


def test_resume_greenlet():
    gr = greenlet.greenlet(resume_greenlet_fn)
    gr.switch(10)
    serframe = skt.copy_frame_from_greenlet(gr, serialize=True)
    capsule = skt.deserialize_frame(serframe)
    gr2 = greenlet.greenlet(skt.run_frame)
    result = gr2.switch(capsule)
    assert result == 15
    print("Test 'resume_greenlet' passed")


def _write_checkpoint_module(module_dir, module_name, env_key):
    module_path = os.path.join(module_dir, f"{module_name}.py")
    module_source = textwrap.dedent(
        f"""
        import os
        import greenlet

        os.environ['{env_key}'] = 'set'

        def checkpoint(value):
            greenlet.getcurrent().parent.switch()
            return value + 1
        """
    )
    with open(module_path, "w", encoding="utf-8") as f:
        f.write(module_source)


def test_capture_module_source_default_reconstruct():
    env_key = f"SAUERKRAUT_CAPTURE_DEFAULT_{uuid.uuid4().hex}"
    module_name = f"skt_capture_default_{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as temp_dir:
        _write_checkpoint_module(temp_dir, module_name, env_key)
        sys.path.insert(0, temp_dir)
        try:
            module = importlib.import_module(module_name)
            gr = greenlet.greenlet(module.checkpoint)
            gr.switch(10)
            frame_bytes = skt.copy_frame_from_greenlet(
                gr, None, 1, True, True, False, True
            )

            os.environ.pop(env_key, None)
            sys.modules.pop(module_name, None)
            sys.path.remove(temp_dir)

            result = skt.deserialize_frame(frame_bytes, run=True)
            assert result == 11
            assert os.environ.get(env_key) == "set"
            print("Test 'capture_module_source_default_reconstruct' passed")
        finally:
            if temp_dir in sys.path:
                sys.path.remove(temp_dir)
            sys.modules.pop(module_name, None)
            os.environ.pop(env_key, None)


def test_capture_module_source_reconstruct_disabled():
    env_key = f"SAUERKRAUT_CAPTURE_OFF_{uuid.uuid4().hex}"
    module_name = f"skt_capture_disabled_{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as temp_dir:
        _write_checkpoint_module(temp_dir, module_name, env_key)
        sys.path.insert(0, temp_dir)
        try:
            module = importlib.import_module(module_name)
            gr = greenlet.greenlet(module.checkpoint)
            gr.switch(10)
            frame_bytes = skt.copy_frame_from_greenlet(
                gr, None, 1, True, True, False, True
            )

            os.environ.pop(env_key, None)
            result = skt.deserialize_frame(frame_bytes, run=True, reconstruct_module=False)
            assert result == 11
            assert os.environ.get(env_key) is None
            print("Test 'capture_module_source_reconstruct_disabled' passed")
        finally:
            if temp_dir in sys.path:
                sys.path.remove(temp_dir)
            sys.modules.pop(module_name, None)
            os.environ.pop(env_key, None)


def test_liveness_basic():
    def sample_fn():
        a = 1
        b = 2
        c = a + b
        return c

    code = sample_fn.__code__
    analysis = liveness.LivenessAnalysis(code)
    offsets = analysis.get_offsets()
    assert len(offsets) > 0

    for offset in offsets:
        live = analysis.get_live_variables_at_offset(offset)
        dead = analysis.get_dead_variables_at_offset(offset)
        assert len(live & dead) == 0
    print("Test 'liveness_basic' passed")


def test_liveness_dead_variables():
    def fn_with_dead():
        x = 1
        y = 2
        z = y + 1
        return z

    code = fn_with_dead.__code__
    analysis = liveness.LivenessAnalysis(code)
    found_x_dead = any(
        "x" in analysis.get_dead_variables_at_offset(o) for o in analysis.get_offsets()
    )
    assert found_x_dead
    print("Test 'liveness_dead_variables' passed")


def test_liveness_module_function():
    def cached_fn():
        a = 10
        b = 20
        return b

    code = cached_fn.__code__
    analysis = liveness.LivenessAnalysis(code)
    offset = analysis.get_offsets()[0]
    dead1 = liveness.get_dead_variables_at_offset(code, offset)
    dead2 = liveness.get_dead_variables_at_offset(code, offset)
    assert dead1 == dead2
    print("Test 'liveness_module_function' passed")


def test_liveness_invalid_offset():
    def simple_fn():
        return 1

    analysis = liveness.LivenessAnalysis(simple_fn.__code__)
    try:
        analysis.get_live_variables_at_offset(99999)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("Test 'liveness_invalid_offset' passed")


def test_liveness_loop():
    def loop_fn():
        total = 0
        for i in range(5):
            total += i
        return total

    analysis = liveness.LivenessAnalysis(loop_fn.__code__)
    offsets = analysis.get_offsets()
    assert len(offsets) > 0
    print("Test 'liveness_loop' passed")


def test_liveness():
    test_liveness_basic()
    test_liveness_dead_variables()
    test_liveness_module_function()
    test_liveness_invalid_offset()
    test_liveness_loop()


test_copy_then_serialize()
test_combined_copy_serialize()
test_for_loop()
test_greenlet()
test_replace_locals()
test_exclude_locals()
test_copy_frame()
test_resume_greenlet()
test_capture_module_source_default_reconstruct()
test_capture_module_source_reconstruct_disabled()
test_liveness()
