import sauerkraut as skt
import greenlet
import numpy as np
import importlib
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid

calls = 0
try:
    import cupy as cp
except Exception:
    cp = None


def _get_cupy_with_gpu_or_none():
    if cp is None:
        print("Skipping CuPy tests: CuPy is not installed")
        return None

    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
    except Exception:
        print("Skipping CuPy tests: failed to query GPU count")
        return None

    if gpu_count < 1:
        print("Skipping CuPy tests: no GPUs available")
        return None
    return cp


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


def test_capture_module_source_cross_file():
    env_key = f"SAUERKRAUT_CAPTURE_CROSS_{uuid.uuid4().hex}"
    module_name = f"skt_capture_cross_{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as temp_dir:
        frame_path = os.path.join(temp_dir, "frame.bin")
        module_path = os.path.join(temp_dir, f"{module_name}.py")
        producer_path = os.path.join(temp_dir, "producer.py")
        consumer_path = os.path.join(temp_dir, "consumer.py")

        module_source = textwrap.dedent(
            f"""
            import os
            import greenlet

            os.environ['{env_key}'] = 'set'

            def checkpoint(value):
                token = "checkpointed"
                greenlet.getcurrent().parent.switch()
                return value + len(token)
            """
        )
        with open(module_path, "w", encoding="utf-8") as f:
            f.write(module_source)

        producer_source = textwrap.dedent(
            f"""
            import importlib
            import pathlib
            import sys
            import greenlet
            import sauerkraut as skt

            sys.path.insert(0, r"{temp_dir}")
            module = importlib.import_module({module_name!r})
            gr = greenlet.greenlet(module.checkpoint)
            gr.switch(30)
            frame_bytes = skt.copy_frame_from_greenlet(
                gr, serialize=True, capture_module_source=True
            )
            pathlib.Path(r"{frame_path}").write_bytes(frame_bytes)
            """
        )
        with open(producer_path, "w", encoding="utf-8") as f:
            f.write(producer_source)

        consumer_source = textwrap.dedent(
            f"""
            import os
            import pathlib
            import sauerkraut as skt

            os.environ.pop('{env_key}', None)
            frame_bytes = pathlib.Path(r"{frame_path}").read_bytes()
            result = skt.deserialize_frame(frame_bytes, run=True)
            assert result == 42
            assert os.environ.get('{env_key}') == 'set'
            """
        )
        with open(consumer_path, "w", encoding="utf-8") as f:
            f.write(consumer_source)

        env = dict(os.environ)

        producer = subprocess.run(
            [sys.executable, producer_path], env=env, capture_output=True, text=True
        )
        assert producer.returncode == 0, (
            "Producer failed\nstdout:\n"
            + producer.stdout
            + "\nstderr:\n"
            + producer.stderr
        )

        os.remove(module_path)

        consumer = subprocess.run(
            [sys.executable, consumer_path], env=env, capture_output=True, text=True
        )
        assert consumer.returncode == 0, (
            "Consumer failed\nstdout:\n"
            + consumer.stdout
            + "\nstderr:\n"
            + consumer.stderr
        )
        print("Test 'capture_module_source_cross_file' passed")


def test_liveness_basic():
    from sauerkraut import liveness

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
    from sauerkraut import liveness

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
    from sauerkraut import liveness

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
    from sauerkraut import liveness

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
    from sauerkraut import liveness

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


def _cupy_local_roundtrip_fn(base):
    assert cp is not None

    arr = cp.arange(6, dtype=cp.float32).reshape(2, 3) + base
    source_device = arr.device.id
    greenlet.getcurrent().parent.switch()
    assert isinstance(arr, cp.ndarray)
    assert arr.device.id == source_device
    arr = arr + 2
    return float(cp.asnumpy(arr).sum())


def test_cupy_local_roundtrip():
    if _get_cupy_with_gpu_or_none() is None:
        return

    gr = greenlet.greenlet(_cupy_local_roundtrip_fn)
    gr.switch(3.0)
    serframe = skt.copy_frame_from_greenlet(gr, serialize=True)
    result = skt.deserialize_frame(serframe, run=True)
    expected = float(cp.asnumpy(cp.arange(6, dtype=cp.float32).reshape(2, 3) + 5.0).sum())
    assert result == expected
    print("Test 'cupy_local_roundtrip' passed")


def _cupy_stack_roundtrip_fn():
    assert cp is not None
    tmp = cp.arange(4, dtype=cp.int32).reshape(2, 2) + greenlet.getcurrent().parent.switch(10)
    return int(cp.asnumpy(tmp).sum())


def test_cupy_stack_roundtrip():
    if _get_cupy_with_gpu_or_none() is None:
        return

    gr = greenlet.greenlet(_cupy_stack_roundtrip_fn)
    gr.switch()
    serframe = skt.copy_frame_from_greenlet(gr, serialize=True)
    capsule = skt.deserialize_frame(serframe)
    assert capsule is not None
    print("Test 'cupy_stack_roundtrip' passed")


class _FakeGpuObject:
    def __dlpack_device__(self):
        return (2, 0)


def _unsupported_gpu_backend_fn():
    fake_gpu_obj = _FakeGpuObject()
    _ = fake_gpu_obj
    return skt.copy_current_frame(serialize=True, exclude_dead_locals=False)


def test_gpu_unsupported_backend_fails():
    try:
        _unsupported_gpu_backend_fn()
        assert False, "Expected RuntimeError for unsupported GPU backend"
    except RuntimeError:
        pass
    print("Test 'gpu_unsupported_backend_fails' passed")


def run_standard_tests():
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
    test_capture_module_source_cross_file()
    test_liveness()


def run_gpu_tests():
    if _get_cupy_with_gpu_or_none() is None:
        return

    test_cupy_local_roundtrip()
    test_cupy_stack_roundtrip()
    test_gpu_unsupported_backend_fails()


run_standard_tests()
run_gpu_tests()
