"""Isolated CPU inference for a pinned TorchScript artifact and numeric arrays."""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = torch.jit.load(str(args.model), map_location='cpu').eval()
    data = torch.from_numpy(np.load(args.input, allow_pickle=False).astype(np.float32))
    parts = []
    with torch.no_grad():
        for chunk in data.split(2048):
            output = model(chunk)
            if output.ndim == 3:
                output = output.mean(dim=1)
            parts.append(output[:, 0].numpy() * .05)
    np.save(args.output, np.concatenate(parts), allow_pickle=False)


if __name__ == '__main__':
    main()
