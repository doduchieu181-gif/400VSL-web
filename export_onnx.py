"""Xuất mô hình SDA-GCN sang ONNX để chạy trong trình duyệt (bản web tĩnh).

Chạy trong D:\\400VSL, đã kích hoạt .venv:
    pip install onnx onnxruntime
    python -X utf8 export_onnx.py

Kết quả (nằm cạnh file này):
    sdagcn.onnx        mô hình cho trình duyệt
    onnx_check.json    đầu ra mẫu để kiểm tra bản JavaScript cho kết quả giống Python
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webui"))

from backend.config import MODEL_ARGS, MODEL_CLASS, WEIGHTS_PATH  # noqa: E402
from backend.pipeline.feeder import to_model_input  # noqa: E402
from backend.pipeline.fusion import fuse  # noqa: E402
from backend.services.catalog import Catalog  # noqa: E402
from einops import rearrange, repeat  # noqa: E402

import model.model28.SDAGCN as sdagcn  # noqa: E402


def get_graph_feature(self, x, k, idx=None):
    """Giống bản gốc, chỉ thay x.get_device() bằng x.device để chạy được trên CPU."""
    N, C, V = x.size()
    if idx is None:
        idx = self.knn(x, k=k)
    idx_base = torch.arange(0, N, device=x.device).view(-1, 1, 1) * V
    idx = (idx + idx_base).view(-1)
    x = rearrange(x, 'n c v -> n v c')
    feature = rearrange(x, 'n v c -> (n v) c')[idx, :]
    feature = feature.view(N, V, k, C)
    x = repeat(x, 'n v c -> n v k c', k=k)
    feature = torch.cat((feature - x, x), dim=3)
    return rearrange(feature, 'n v k c -> n c v k')


sdagcn.EdgeConv.get_graph_feature = get_graph_feature


def load_model():
    __import__(MODEL_ARGS["graph"].rsplit(".", 1)[0])
    model = sdagcn.Model(**MODEL_ARGS)
    weights = torch.load(str(WEIGHTS_PATH), map_location="cpu")
    weights = OrderedDict((k.split("module.")[-1], v) for k, v in weights.items())
    model.load_state_dict(weights)
    return model.eval()


def main():
    model = load_model()
    dummy = torch.zeros(1, 3, 64, MODEL_ARGS["num_point"], 1)
    out_path = ROOT / "sdagcn.onnx"
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["input"], output_names=["logits"],
        opset_version=17, do_constant_folding=True,
    )
    print(f"Đã ghi {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Kiểm tra: ONNX phải cho cùng kết quả với PyTorch trên vài video mẫu thật
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    catalog = Catalog()
    checks = []
    worst = 0.0
    for label in (1, 2, 3, 50, 123, 400):
        raw = catalog.reference_keypoints(label)
        x = to_model_input(fuse(raw, 64), 64)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()[0]
        got = sess.run(None, {"input": x})[0][0]
        diff = float(np.abs(ref - got).max())
        worst = max(worst, diff)
        top = int(np.argmax(ref)) + 1
        print(f"  bài {label:3d}: mô hình đoán bài {top:3d}, lệch PyTorch/ONNX = {diff:.2e}")
        checks.append({"label": label, "logits": [round(float(v), 5) for v in ref],
                       "input_sum": float(x.sum())})
    (ROOT / "onnx_check.json").write_text(json.dumps(checks), encoding="utf-8")
    print("XONG. Lệch lớn nhất:", f"{worst:.2e}", "(nhỏ hơn 1e-3 là tốt)" if worst < 1e-3 else "(LỚN, báo lại)")
    print("Gửi 2 file sdagcn.onnx và onnx_check.json vào khung chat.")


if __name__ == "__main__":
    main()
