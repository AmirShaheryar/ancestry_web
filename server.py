"""Backend for the ancestry explorer: loads the saved models and serves a small JSON API
plus the static frontend in ./static."""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"

# macro F1 per fold, copied from the 5-fold cross-validation cell in the notebook
CV = {
    "Random Forest": [0.9437, 0.9677, 0.9697, 0.9657, 0.9778],
    "ANN": [0.9278, 0.9528, 0.9373, 0.9518, 0.9497],
}

scaler = joblib.load(MODEL_DIR / "scaler.joblib")
pca = joblib.load(MODEL_DIR / "pca.joblib")
rf = joblib.load(MODEL_DIR / "random_forest.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
train_pcs = np.load(MODEL_DIR / "train_pcs.npy")
train_labels = np.load(MODEL_DIR / "train_labels.npy", allow_pickle=True)
demo_x = np.load(MODEL_DIR / "demo_samples.npy")
demo_y = np.load(MODEL_DIR / "demo_labels.npy", allow_pickle=True)

classes = [str(c) for c in encoder.classes_]
n_pcs = train_pcs.shape[1]


def load_ann():
    """Prefer the tiny NumPy weights (no TensorFlow). Fall back to the .keras file."""
    npz = MODEL_DIR / "ann_weights.npz"
    if npz.exists():
        w = dict(np.load(npz))

        def forward(x):
            h = np.maximum(x @ w["W1"] + w["b1"], 0)
            h = np.maximum(h @ w["W2"] + w["b2"], 0)
            z = h @ w["W3"] + w["b3"]
            e = np.exp(z - z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        return forward

    keras_file = MODEL_DIR / "ann_model.keras"
    if keras_file.exists():
        from tensorflow import keras
        model = keras.models.load_model(keras_file)
        return lambda x: model.predict(x, verbose=0)
    return None


ann = load_ann()
model_names = ["Random Forest"] + (["ANN"] if ann is not None else [])


def run_models(pcs):
    """pcs: one sample in PCA space. Returns {model: {class: probability}}."""
    x = np.asarray(pcs, dtype=np.float32).reshape(1, -1)
    out = {"Random Forest": dict(zip(map(str, rf.classes_), rf.predict_proba(x)[0].tolist()))}
    if ann is not None:
        out["ANN"] = dict(zip(classes, ann(x)[0].tolist()))
    return out


app = FastAPI(title="Ancestry Explorer")


@app.get("/api/meta")
def meta():
    return {
        "classes": classes,
        "models": model_names,
        "n_samples": len(demo_x),
        "n_pcs": n_pcs,
        "n_snps": int(scaler.n_features_in_),
        "n_train": len(train_pcs),
        "variance": (pca.explained_variance_ratio_ * 100).round(2).tolist(),
        "pc_range": [[float(train_pcs[:, i].min()), float(train_pcs[:, i].max())] for i in range(5)],
        "cv": CV,
    }


@app.get("/api/train")
def train():
    """First 6 PCs of every training person, for the background scatter plot."""
    return {"pcs": np.round(train_pcs[:, :6], 3).tolist(), "labels": train_labels.astype(str).tolist()}


@app.get("/api/sample/{index}")
def sample(index: int):
    if not 0 <= index < len(demo_x):
        raise HTTPException(404, "No such demo sample")
    genotype = demo_x[index].reshape(1, -1).astype(np.float32)
    pcs = pca.transform(scaler.transform(genotype))[0]
    return {"index": index, "truth": str(demo_y[index]), "pcs": pcs.tolist(),
            "predictions": run_models(pcs)}


class PCsRequest(BaseModel):
    pcs: list[float]


@app.post("/api/predict")
def predict(req: PCsRequest):
    """Predict from PCA coordinates directly (used by the playground sliders)."""
    if len(req.pcs) > n_pcs:
        raise HTTPException(400, f"Expected at most {n_pcs} values")
    pcs = req.pcs + [0.0] * (n_pcs - len(req.pcs))
    return {"predictions": run_models(pcs)}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000)
