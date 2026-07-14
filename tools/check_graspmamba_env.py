"""Check the optional GraspMamba runtime before starting a long experiment."""

import argparse
from pathlib import Path


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
    model = create_model("mamba_vision_T", pretrained=False, num_classes=0)
    channels = [80, 160, 320, 640]
    print(f"MambaVision-T: OK, expected stage channels={channels}")
    del model

    clip_path = Path(args.clip)
    print(f"CLIP: {'OK' if clip_path.is_file() else 'MISSING'} ({clip_path.resolve()})")
    mamba_path = Path(args.mamba)
    state = "OK" if mamba_path.is_file() else "missing; first model build will download it"
    print(f"MambaVision checkpoint: {state} ({mamba_path.resolve()})")


if __name__ == "__main__":
    main()
