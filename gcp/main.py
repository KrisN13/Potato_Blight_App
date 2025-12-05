# main.py
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import functions_framework
import os, io, json, base64, traceback, sys

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BUCKET_NAME      = "potato-model-tf-classification"   # your GCS bucket
REMOTE_MODEL_DIR = "models"                            # folder with saved_model.pb + variables/
LOCAL_MODEL_DIR  = "/tmp/potato_saved_model"           # local cache
# Make sure this order matches training (e.g., image_dataset_from_directory alphabetical order)
CLASS_NAMES      = ["Early Blight", "Late Blight", "Healthy",]

# ── GLOBALS (cached across invocations) ────────────────────────────────────────
infer_fn   = None
input_key  = None
input_dtype = None
input_shape = None
output_key  = None

# ── UTIL ───────────────────────────────────────────────────────────────────────
def jsonify(obj, status=200, ct="application/json"):
    return json.dumps(obj), status, {"Access-Control-Allow-Origin": "*", "Content-Type": ct}

def download_model_dir(bucket_name: str, prefix: str, local_dir: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix.rstrip("/") + "/")
    os.makedirs(local_dir, exist_ok=True)
    count = 0
    for b in blobs:
        if b.name.endswith("/"):
            continue
        rel = os.path.relpath(b.name, prefix)
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        b.download_to_filename(dst)
        count += 1
        print(f"[download] {b.name} -> {dst}")
    print(f"[download] downloaded {count} files from gs://{bucket_name}/{prefix}/")

def ensure_model_loaded() -> None:
    """Load TF SavedModel once and wire its serving signature."""
    global infer_fn, input_key, input_dtype, input_shape, output_key
    if infer_fn is not None:
        return
    download_model_dir(BUCKET_NAME, REMOTE_MODEL_DIR, LOCAL_MODEL_DIR)
    loaded = tf.saved_model.load(LOCAL_MODEL_DIR)                 # Keras 3-safe loader
    sig = loaded.signatures["serving_default"]
    infer_fn = sig
    # capture input/output signatures so we feed exactly what the model expects
    _, input_spec = sig.structured_input_signature
    input_key = next(iter(input_spec.keys()))
    input_tensor_spec = input_spec[input_key]
    input_dtype = input_tensor_spec.dtype                         # tf.float32 or tf.uint8, etc.
    input_shape = input_tensor_spec.shape                         # (None, H, W, 3)
    output_key = next(iter(sig.structured_outputs.keys()))
    print(f"[signature] input_key={input_key} dtype={input_dtype} shape={input_shape} output_key={output_key}")

def preprocess(pil: Image.Image, scale_mode: str | None) -> tf.Tensor:
    """
    Resize to the model's spatial size and scale pixels.
    Default = '255f': float32 in [0,255] (matches your training pipeline).
    """
    scale_mode = (scale_mode or "255f").strip().lower()
    h = input_shape[1] if input_shape.rank >= 3 and input_shape[1] is not None else 256
    w = input_shape[2] if input_shape.rank >= 3 and input_shape[2] is not None else 256
    arr = np.array(pil.convert("RGB").resize((w, h)))

    if scale_mode in ("255", "255f"):            # float32 0..255  ← your training setup
        arr = arr.astype("float32")
    elif scale_mode == "01":                     # float32 0..1
        arr = arr.astype("float32") / 255.0
    elif scale_mode == "-1to1":                  # float32 [-1,1]
        arr = (arr.astype("float32") / 127.5) - 1.0
    else:                                        # fallback: float32 0..255
        arr = arr.astype("float32")

    batch = tf.expand_dims(arr, 0)
    # quick breadcrumb for sanity (visible in Cloud Logs)
    print(f"[preprocess] mode={scale_mode} shape={batch.shape} dtype={batch.dtype} "
          f"min={float(tf.reduce_min(batch))} max={float(tf.reduce_max(batch))}")
    return batch

def postprocess_probs(row: np.ndarray) -> np.ndarray:
    """Softmax logits if needed; leave as-is if already probabilities."""
    if not np.allclose(np.sum(row), 1.0, atol=1e-3) or np.any(row < 0):
        e = np.exp(row - np.max(row))
        row = e / np.sum(e)
    return row

# ── HTTP ENTRYPOINT ────────────────────────────────────────────────────────────
@functions_framework.http
def predict(request):
    # CORS preflight
    if request.method == "OPTIONS":
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    # simple form for manual testing
    if request.method == "GET":
        html = """<!doctype html><html lang="en">
        <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Potato Blight Classifier</title>
        <body style="font-family:system-ui,Segoe UI,Arial;margin:2rem;">
        <h3>Potato Blight Classifier</h3>
        <form method="post" enctype="multipart/form-data" action="/predict">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Classify</button>
        </form>
        <p style="opacity:.6;margin-top:1rem;">Upload a leaf image (JPG/PNG). Returns JSON.</p>
        </body></html>"""
        return (html, 200, {"Access-Control-Allow-Origin": "*", "Content-Type": "text/html"})


    try:
        ensure_model_loaded()

        # optional params
        scale_mode = (request.args.get("scale") or request.form.get("scale") or None)
        debug = (request.args.get("debug") or request.form.get("debug") or "0").strip() == "1"

        # read image (multipart or JSON base64)
        if request.files and "file" in request.files:
            pil = Image.open(request.files["file"].stream)
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            b64 = data.get("image_b64")
            if not b64:
                return jsonify({"error": "Missing 'file' or 'image_b64'."}, 400)
            pil = Image.open(io.BytesIO(base64.b64decode(b64)))
        else:
            return jsonify({"error": "No file uploaded. Send multipart/form-data with key 'file'."}, 400)

        # preprocess → infer → postprocess
        batch = preprocess(pil, scale_mode)
        outputs = infer_fn(**{input_key: batch})
        preds = outputs[output_key].numpy()
        row = postprocess_probs(preds[0])

        probs = [float(x) for x in row.tolist()]
        idx = int(np.argmax(probs))
        resp = {
            "class": CLASS_NAMES[idx],
            "confidence": round(float(max(probs) * 100.0), 2),
            "probs": {CLASS_NAMES[i]: round(float(p * 100.0), 2) for i, p in enumerate(probs)},
        }
        if debug:
            resp.update({
                "signature": {
                    "input_key": input_key,
                    "dtype": str(input_dtype),
                    "shape": tuple(int(d) if d is not None else None for d in input_shape),
                    "output_key": output_key,
                },
                "scaling": (scale_mode or "255f"),
                "probs_raw": probs,
            })
        return jsonify(resp, 200)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}, 500)
