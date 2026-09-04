"""Check the optional GraspMamba runtime before starting a long experiment."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pretrained import ensure_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="pretrain/RN50.pt")
    parser.add_argument(
        "--mamba", default="pretrain/mambavision_tiny_1k.pth.tar"
    )
    args = parser.parse_args()

    import torch

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"torch CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import mamba_ssm
        from mambavision import create_model
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "MambaVision import failed. Install it after the correct CUDA PyTorch "
            "build with: pip install -r requirement-mamba.txt\n"
            f"Original error: {exc}"
        )

    print(f"mamba_ssm: {getattr(mamba_ssm, '__version__', 'unknown')}")
    mamba_path = ensure_pretrained(args.mamba, "mambavision-t")
    from model.graspmamba import _official_checkpoint_context

    with _official_checkpoint_context(mamba_path, "mambavision-t"):
        model = create_model(
            "mamba_vision_T",
            pretrained=True,
            model_path=str(mamba_path),
            num_classes=0,
        )
    channels = [80, 160, 320, 640]
    print(
        "MambaVision-T: OK, official checkpoint loaded; "
        f"expected stage channels={channels}"
    )
    del model

    clip_path = Path(args.clip)
    print(f"CLIP: {'OK' if clip_path.is_file() else 'MISSING'} ({clip_path.resolve()})")
    print(f"MambaVision checkpoint: OK ({mamba_path.resolve()})")


if __name__ == "__main__":
    main()
