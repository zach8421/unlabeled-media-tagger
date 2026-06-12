"""GPU smoke test — guards the pinned TF/CUDA stack (requirements-gpu.txt).

Skips cleanly when TensorFlow is unavailable or no GPU is registered, so the
suite stays green on CPU-only / CI machines. When a GPU *is* present it runs a
real matmul on the device — the same GATE 1 check from the GPU enablement work
— so a broken pin set fails loudly instead of silently falling back to CPU.

Two distinct failure modes this catches:
  - dlopen regression: the device never registers (caught by the skipif turning
    into an unexpected skip if you expected a GPU — and by CI config asserting
    the test ran).
  - missing Pascal cubin: the device registers but a kernel launch raises "no
    kernel image is available for execution on the device" — caught by forcing
    materialization with .numpy() below.

See dev-log/2026-06-11-gpu-enablement.md.
"""

import pytest

tf = pytest.importorskip("tensorflow")

pytestmark = pytest.mark.gpu


def _gpus():
    return tf.config.list_physical_devices("GPU")


@pytest.mark.skipif(not _gpus(), reason="no GPU registered (CPU-only environment)")
def test_gpu_matmul_runs_on_device():
    """A real matmul must execute on the GPU, not just register the device.

    A device can register and still fail at kernel launch if the wheel lacks an
    sm_61 cubin for Pascal, so we assert the op is placed on the GPU and force
    materialization to surface any kernel-launch failure here.
    """
    with tf.device("/GPU:0"):
        a = tf.random.normal([512, 512])
        b = tf.random.normal([512, 512])
        c = tf.matmul(a, b)

    assert "GPU" in c.device
    value = c.numpy()  # materialize — a missing sm_61 cubin raises here
    assert value.shape == (512, 512)
