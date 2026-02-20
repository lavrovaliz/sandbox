import argparse
import os
from pathlib import Path

import numpy as np
import pydicom
import torch


def run_inference(dicom_dir: Path, model_path: Path, out_dir: Path, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    model = torch.load(model_path)  
    model.eval().to(device)

    for fname in os.listdir(dicom_dir):  
        dcm_path = dicom_dir / fname
        if dcm_path.is_dir():
            continue

        ds = pydicom.dcmread(str(dcm_path))
        img = ds.pixel_array  

        X = img.astype(np.float32) / 255.0  
        X = torch.tensor(X).unsqueeze(0).unsqueeze(0).to(device)

        pred = model(X)  
        maskOutput = (pred > 0.5).float().cpu().numpy()

        np.save(out_dir / f"{dcm_path.stem}.npy", maskOutput)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_inference(
        Path(args.dicom_dir),
        Path(args.model_path),
        Path(args.out_dir),
        args.device,
    )


if __name__ == "__main__":
    main()
