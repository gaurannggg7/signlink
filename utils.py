import os
import torch

def get_best_device() -> torch.device:
    """
    Return the fastest available device in this order:
      1) TPU (if on Colab with TPU runtime)
      2) CUDA GPU
      3) Apple MPS
      4) CPU
    """
    if 'COLAB_TPU_ADDR' in os.environ:
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            pass

    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
