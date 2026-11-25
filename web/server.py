"""Web lokal untuk membandingkan segmentasi instance pada satu frame."""
import argparse
import base64
import csv
import io
import json
import threading
import time
from functools import lru_cache
from importlib.util import find_spec
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
RESULTS = ROOT / "results"
MODELS = {"yolov8n-seg": "YOLOv8n-seg", "yolov8s-seg": "YOLOv8s-seg",
          "maskrcnn_resnet50_fpn": "Mask R-CNN"}
SIZES = (320, 640, 960)
LOCK = threading.Lock()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checkpoint(runs, model, size):
    folder = runs / f"{model}_{size}"
    return folder / ("weights/best.pt" if model.startswith("yolo") else "best.pt")


def catalog(runs):
    with (RESULTS / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    stats = {}
    for row in rows:
        name, size = row["model"], int(row["input_size"])
        folder = RESULTS / "runs" / f"{name}_{size}"
        entry = {"ready": checkpoint(runs, name, size).is_file(),
                 "delta": float(row["domain_shift_AP_points"]),
                 "minutes": float(row["elapsed_minutes"]), "vram": float(row["peak_vram_GiB"])}
        for split, key in (("test", "common_2"), ("personal", "all_annotated")):
            values = read(folder / f"{split}_metrics.json")[key]
            entry[split] = {k: values[k] for k in ("AP", "AP50", "AP75", "per_class_AP")}
        stats[folder.name] = entry
    audit = read(RESULTS / "audit_coco.json")
    dataset = {
        "splits": {key: audit[key] for key in ("train", "valid", "test", "personal")},
        "categories": read(RESULTS / "categories.json"),
        "comparison_ids": read(RESULTS / "domain_shift_policy.json")["category_ids"],
    }
    return {"models": MODELS, "stats": stats, "dataset": dataset}


@lru_cache(maxsize=3)
def load(path, model, size, device):
    import torch
    if model.startswith("yolo"):
        from ultralytics import YOLO
        net = YOLO(str(path))
        if net.task != "segment" or len(net.names) != 74:
            raise ValueError("Checkpoint YOLO harus berasal dari training 74 kelas.")
        return net
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    net = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=75,
                                min_size=size, max_size=size, box_score_thresh=.001,
                                box_nms_thresh=.7, box_detections_per_img=100)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return net.eval().to(device)


def infer(photo, model, size, conf, path):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torchvision.transforms.functional import pil_to_tensor

    labels = {c["id"]: c["name"] for c in read(RESULTS / "categories.json")}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load(path, model, size, device)
    w, h = photo.size
    nw, nh = [max(1, round(v * size / max(w, h))) for v in (w, h)]
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(photo.resize((nw, nh), Image.Resampling.BILINEAR), (x, y))
    start = time.perf_counter()
    with torch.inference_mode():
        if model.startswith("yolo"):
            result = net.predict(canvas, imgsz=size, conf=.001, iou=.7, max_det=100,
                                 retina_masks=True, device=device, verbose=False)[0]
            ids = result.boxes.cls.int().cpu().tolist()
            names = [net.names[i] for i in ids]
            scores = result.boxes.conf.cpu().tolist()
            masks = result.masks.data if result.masks is not None else []
        else:
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                result = net([pil_to_tensor(canvas).to(device).float() / 255])[0]
            names = [labels[i] for i in result["labels"].cpu().tolist()]
            scores = result["scores"].cpu().tolist()
            masks = result["masks"][:, 0]
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000
        image = np.array(canvas)
        items = []
        palette = [(0, 165, 225), (255, 185, 42), (64, 204, 153), (216, 113, 203)]
        for name, score, mask in zip(names, scores, masks):
            if score < conf:
                continue
            area = (mask >= .5).cpu().numpy()
            rows, cols = np.where(area[y:y+nh, x:x+nw])
            if not len(rows):
                continue
            color = palette[len(items) % len(palette)]
            image[area] = image[area] * .5 + np.array(color) * .5
            mid = len(rows) // 2
            items.append((name, score, int(cols[mid]), int(rows[mid])))
    output = Image.fromarray(image).crop((x, y, x+nw, y+nh))
    draw = ImageDraw.Draw(output)
    for name, score, px, py in items:
        label = f"{name} {score:.0%}"
        px = max(0, min(px, nw-int(draw.textlength(label))-4))
        py = max(0, min(py, nh-16))
        draw.rectangle(draw.textbbox((px, py), label), fill="#102947")
        draw.text((px, py), label, fill="white")
    buf = io.BytesIO()
    output.save(buf, "JPEG", quality=89)
    return {"image": base64.b64encode(buf.getvalue()).decode(), "ms": round(ms),
            "device": device.upper(), "count": len(items)}


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, mime="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/models":
            return self.send(200, catalog(self.server.runs))
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "application/javascript"), "/style.css": ("style.css", "text/css")}
        if self.path not in files:
            return self.send(404, {"error": "Halaman tidak ditemukan."})
        name, mime = files[self.path]
        self.send(200, (WEB / name).read_bytes(), mime)

    def do_POST(self):
        url = urlsplit(self.path)
        if url.path != "/api/infer":
            return self.send(404, {"error": "Alamat tidak ditemukan."})
        try:
            args = parse_qs(url.query)
            model, size, conf = args["model"][0], int(args["size"][0]), float(args["conf"][0])
            length = int(self.headers.get("Content-Length", "0"))
            if model not in MODELS or size not in SIZES or not .001 <= conf <= 1:
                raise ValueError("Pilihan model, ukuran, atau threshold tidak sesuai.")
            if self.headers.get_content_type() != "image/jpeg" or not 0 < length <= 12_000_000:
                raise ValueError("Kirim satu frame JPEG maksimal 12 MB.")
            path = checkpoint(self.server.runs, model, size)
            if not path.is_file():
                return self.send(409, {"error": f"Checkpoint belum tersedia: {path}"})
            required = {"PIL": "Pillow", "numpy": "NumPy", "torch": "PyTorch",
                        "torchvision": "Torchvision"}
            if model.startswith("yolo"):
                required["ultralytics"] = "Ultralytics"
            missing = [package for module, package in required.items() if find_spec(module) is None]
            if missing:
                packages = ", ".join(missing)
                return self.send(503, {
                    "error": f"Paket inferensi belum terpasang: {packages}. Jalankan server dengan environment Python 3.12 proyek."
                })
            from PIL import Image
            try:
                with Image.open(io.BytesIO(self.rfile.read(length))) as source:
                    if source.width * source.height > 5_000_000:
                        raise ValueError("Resolusi gambar terlalu besar.")
                    photo = source.convert("RGB")
            except (OSError, Image.DecompressionBombError):
                return self.send(400, {"error": "Gambar JPEG tidak bisa dibaca."})
            with LOCK:
                data = infer(photo, model, size, conf, path)
        except (ValueError, KeyError) as exc:
            return self.send(400, {"error": str(exc)})
        except Exception as exc:
            return self.send(500, {"error": f"Inferensi gagal: {exc}"})
        self.send(200, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    options = parser.parse_args()
    candidates = [ROOT / "work/runs", ROOT.parent / "Week11/Submission_5054241001/runs",
                  Path(r"E:\Kuliah\Semester5\CV\Week11\Submission_5054241001\runs")]
    runs = options.runs or max(candidates, key=lambda p: sum(checkpoint(p,m,s).is_file() for m in MODELS for s in SIZES))
    server = ThreadingHTTPServer(("127.0.0.1", options.port), Handler)
    server.runs = runs
    print(f"Buka http://127.0.0.1:{server.server_port} | Checkpoint: {runs}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
