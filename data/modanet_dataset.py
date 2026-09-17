from pathlib import Path
import re

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


MODULATIONS = [
    "16APSK",
    "16QAM",
    "4PAM",
    "64QAM",
    "8FSK",
    "8PSK",
    "DSB-SC",
    "LFM",
    "PSK",
    "QFSK",
    "QPSK",
    "SSB-SC",
]

MOD_TO_INDEX = {name: idx for idx, name in enumerate(MODULATIONS)}


class MoDANetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir).expanduser().resolve()

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy MoDANet dataset tại: {self.root_dir}"
            )

        self.files = sorted(
            self.root_dir.glob("SNR*dB/*/*/*.mat")
        )

        if not self.files:
            raise RuntimeError(
                f"Không tìm thấy file .mat nào trong: {self.root_dir}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        # --------------------------------------------------
        # 1. Đọc tín hiệu từ file MATLAB
        # data gốc có shape: (1024, 5, 2)
        # --------------------------------------------------
        mat = loadmat(file_path)

        if "data" not in mat:
            raise KeyError(
                f"File không chứa biến 'data': {file_path}"
            )

        x = mat["data"]

        if x.shape != (1024, 5, 2):
            raise ValueError(
                f"Shape không mong đợi {x.shape} tại {file_path}"
            )

        # (1024, 5, 2) -> (2, 5, 1024)
        x = np.transpose(x, (2, 1, 0))
        x = np.ascontiguousarray(x, dtype=np.float32)

        x = torch.from_numpy(x)

        # --------------------------------------------------
        # 2. Đọc nhãn từ cấu trúc thư mục
        #
        # Ví dụ:
        # SNR-07dB/QPSK/+51/data00048308.mat
        # --------------------------------------------------
        relative_parts = file_path.relative_to(self.root_dir).parts

        snr_dir = relative_parts[0]
        modulation_name = relative_parts[1]
        doa_dir = relative_parts[2]

        # SNR-07dB -> -7
        # SNR+20dB -> +20
        match = re.fullmatch(r"SNR([+-]\d+)dB", snr_dir)

        if match is None:
            raise ValueError(
                f"Không đọc được SNR từ thư mục: {snr_dir}"
            )

        snr_db = int(match.group(1))

        # QPSK -> class index
        if modulation_name not in MOD_TO_INDEX:
            raise ValueError(
                f"Modulation không hợp lệ: {modulation_name}"
            )

        mod_label = MOD_TO_INDEX[modulation_name]

        # +51 -> 51 độ
        # -01 -> -1 độ
        doa_deg = int(doa_dir)

        if not -60 <= doa_deg <= 60:
            raise ValueError(
                f"DOA ngoài khoảng [-60, 60]: {doa_deg}"
            )

        # -60 ... +60  ->  0 ... 120
        doa_label = doa_deg + 60

        return {
            "x": x,
            "mod_label": torch.tensor(mod_label, dtype=torch.long),
            "doa_label": torch.tensor(doa_label, dtype=torch.long),
            "doa_deg": doa_deg,
            "snr_db": snr_db,
            "path": str(file_path),
        }
