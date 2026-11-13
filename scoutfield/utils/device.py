"""
Kaggle offers the Tesla P100 (sm_60) while its images ship a PyTorch compiled
for sm_70 and above. The session starts, weights download, the model moves to the GPU,
and the run dies somewhere inside the first forward pass with ``CUDA error: no kernel
image is available for execution on the device`` — a message naming neither the card
nor the fix, reached only after spending GPU quota.

Every entry point that moves a tensor to a device goes through ``select_device``, so
the check cannot be present in training and absent in calibration.
"""

from __future__ import annotations


def select_device(preferred: str | None = None) -> str:
    """Return ``"cuda"`` or ``"cpu"``, refusing a GPU this build cannot target.

    ``preferred`` overrides the choice of device, but a CUDA request is still checked:
    a device that cannot run should be named now, not reported later as a CUDA error.

    The check compares the device's compute capability with the architectures this
    PyTorch build ships. Only a device *older* than every shipped architecture is
    refused, and it is refused here, with an error that names the card and the fix,
    rather than inside the first forward pass. A newer device is allowed, because
    PyTorch can JIT a forward-compatible PTX image for it.
    """
    import torch

    if preferred is not None and preferred != "cuda":
        return preferred
    if not torch.cuda.is_available():
        if preferred == "cuda":
            raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    shipped = sorted(
        int(arch.split("_")[1]) for arch in torch.cuda.get_arch_list()
        if arch.startswith("sm_") and arch.split("_")[1].isdigit()
    )
    if shipped and capability < shipped[0]:
        raise RuntimeError(
            f"{torch.cuda.get_device_name()} has compute capability {major}.{minor} "
            f"(sm_{capability}), but this PyTorch build ships kernels only for "
            f"{', '.join('sm_' + str(s) for s in shipped)}. The run would fail inside "
            f"the first forward pass. Use a GPU with compute capability "
            f"{shipped[0] // 10}.{shipped[0] % 10} or higher; on Kaggle, change the session "
            f"accelerator from P100 to GPU T4 x2 (sm_75) and re-run."
        )
    return "cuda"
