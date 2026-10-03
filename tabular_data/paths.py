"""表格模块的数据、上传文件和运行结果路径。"""

import os
from pathlib import Path

from config import BASE_DIR


TABULAR_DIR = BASE_DIR / "tabular_data"
DATASET_DIR = Path(os.getenv("TABULAR_DATASET_DIR", str(TABULAR_DIR / "datasets"))).resolve()
UPLOAD_DIR = Path(os.getenv("TABULAR_UPLOAD_DIR", str(TABULAR_DIR / "uploads"))).resolve()
OUTPUT_DIR = Path(os.getenv("TABULAR_OUTPUT_DIR", str(TABULAR_DIR / "outputs"))).resolve()
