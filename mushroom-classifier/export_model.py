"""
Export the trained model from `mushroom-classification.ipynb`.

The notebook is NEVER touched and its code is NEVER rewritten: this script
reads the .ipynb file, executes its code cells verbatim and in order, and then
serialises whatever the notebook produced in its own namespace:

    mushroom_model.pth   -> state_dict of the trained MushroomANN
    model_meta.json      -> one-hot encoding scheme (feature -> column mapping)
    encoded_dataset.npz  -> the exact X_encoded matrix + y the notebook built
    insights.json        -> EDA statistics for the website's insights section

Run:  python export_model.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # notebook calls plt.show(); keep it head-less

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).parent
NOTEBOOK = HERE / "mushroom-classification.ipynb"
OUT_DIR = HERE
WEB_DATA = HERE / "static" / "data"


# ── 1. Run the notebook exactly as written ────────────────────────────────────
def run_notebook(path: Path) -> dict:
    nb = json.loads(path.read_text())
    ns: dict = {"__name__": "__notebook__", "__file__": str(path)}
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    for i, cell in enumerate(code_cells, start=1):
        source = "".join(cell["source"])
        if not source.strip():
            continue
        print(f"  [{i:2d}/{len(code_cells)}] running cell...", flush=True)
        exec(compile(source, f"<notebook cell {i}>", "exec"), ns)

    return ns


# ── 2. Presentation metadata (UCI legend, Turkish) ────────────────────────────
FEATURE_LABELS = {
    "cap-shape": ("Şapka Şekli", "🍄"),
    "cap-surface": ("Şapka Yüzeyi", "🪵"),
    "cap-color": ("Şapka Rengi", "🎨"),
    "bruises": ("Çürük / Morarma", "🩹"),
    "odor": ("Koku", "👃"),
    "gill-spacing": ("Lamel Aralığı", "📏"),
    "gill-size": ("Lamel Boyutu", "📐"),
    "gill-color": ("Lamel Rengi", "🌈"),
    "stalk-shape": ("Sap Şekli", "🌱"),
    "stalk-root": ("Sap Kökü", "🌰"),
    "stalk-surface-above-ring": ("Sap Yüzeyi (halka üstü)", "⬆️"),
    "stalk-surface-below-ring": ("Sap Yüzeyi (halka altı)", "⬇️"),
    "stalk-color-above-ring": ("Sap Rengi (halka üstü)", "🖌️"),
    "stalk-color-below-ring": ("Sap Rengi (halka altı)", "🖍️"),
    "veil-color": ("Zar Rengi", "🎭"),
    "ring-number": ("Halka Sayısı", "💍"),
    "ring-type": ("Halka Tipi", "⭕"),
    "spore-print-color": ("Spor İzi Rengi", "✨"),
    "population": ("Topluluk", "👥"),
    "habitat": ("Yaşam Alanı", "🌍"),
}

VALUE_LABELS = {
    "cap-shape": {"b": "Çan", "c": "Konik", "x": "Dışbükey", "f": "Düz",
                  "k": "Tümsekli", "s": "Çukur"},
    "cap-surface": {"f": "Lifli", "g": "Oluklu", "y": "Pullu", "s": "Pürüzsüz"},
    "cap-color": {"n": "Kahverengi", "b": "Devetüyü", "c": "Tarçın", "g": "Gri",
                  "r": "Yeşil", "p": "Pembe", "u": "Mor", "e": "Kırmızı",
                  "w": "Beyaz", "y": "Sarı"},
    "bruises": {"t": "Morarıyor", "f": "Morarmıyor"},
    "odor": {"a": "Badem", "l": "Anason", "c": "Katran", "y": "Balık",
             "f": "Kokuşmuş", "m": "Küf", "n": "Kokusuz", "p": "Keskin",
             "s": "Baharatlı"},
    "gill-spacing": {"c": "Sık", "w": "Çok sık", "d": "Seyrek"},
    "gill-size": {"b": "Geniş", "n": "Dar"},
    "gill-color": {"k": "Siyah", "n": "Kahverengi", "b": "Devetüyü",
                   "h": "Çikolata", "g": "Gri", "r": "Yeşil", "o": "Turuncu",
                   "p": "Pembe", "u": "Mor", "e": "Kırmızı", "w": "Beyaz",
                   "y": "Sarı"},
    "stalk-shape": {"e": "Genişleyen", "t": "Daralan"},
    "stalk-root": {"b": "Soğanlı", "c": "Topuzlu", "u": "Kadeh", "e": "Eşit",
                   "z": "Rizomlu", "r": "Köklü", "?": "Bilinmiyor"},
    "stalk-surface-above-ring": {"f": "Lifli", "y": "Pullu", "k": "İpeksi",
                                 "s": "Pürüzsüz"},
    "stalk-surface-below-ring": {"f": "Lifli", "y": "Pullu", "k": "İpeksi",
                                 "s": "Pürüzsüz"},
    "stalk-color-above-ring": {"n": "Kahverengi", "b": "Devetüyü", "c": "Tarçın",
                               "g": "Gri", "o": "Turuncu", "p": "Pembe",
                               "e": "Kırmızı", "w": "Beyaz", "y": "Sarı"},
    "stalk-color-below-ring": {"n": "Kahverengi", "b": "Devetüyü", "c": "Tarçın",
                               "g": "Gri", "o": "Turuncu", "p": "Pembe",
                               "e": "Kırmızı", "w": "Beyaz", "y": "Sarı"},
    "veil-color": {"n": "Kahverengi", "o": "Turuncu", "w": "Beyaz", "y": "Sarı"},
    "ring-number": {"n": "Yok", "o": "Bir", "t": "İki"},
    "ring-type": {"c": "Ağsı", "e": "Geçici", "f": "Yayvan", "l": "Büyük",
                  "n": "Yok", "p": "Sarkık", "s": "Kılıf", "z": "Kuşak"},
    "spore-print-color": {"k": "Siyah", "n": "Kahverengi", "b": "Devetüyü",
                          "h": "Çikolata", "r": "Yeşil", "o": "Turuncu",
                          "u": "Mor", "w": "Beyaz", "y": "Sarı"},
    "population": {"a": "Bol", "c": "Kümeli", "n": "Çok sayıda",
                   "s": "Dağınık", "v": "Birkaç", "y": "Tek başına"},
    "habitat": {"g": "Çayır", "l": "Yaprak", "m": "Otlak", "p": "Patika",
                "u": "Şehir", "w": "Atık", "d": "Orman"},
}

# swatch colours for the value chips that describe a colour
COLOR_SWATCH = {
    "Kahverengi": "#8b5a2b", "Devetüyü": "#d8c9a3", "Tarçın": "#a0522d",
    "Gri": "#9ca3af", "Yeşil": "#4ade80", "Pembe": "#f9a8d4", "Mor": "#a78bfa",
    "Kırmızı": "#ef4444", "Beyaz": "#f4f4f5", "Sarı": "#facc15",
    "Siyah": "#27272a", "Çikolata": "#5c3a21", "Turuncu": "#fb923c",
}


def main() -> None:
    print(f"→ Executing notebook: {NOTEBOOK.name}")
    ns = run_notebook(NOTEBOOK)

    model = ns["model"]
    df = ns["df"]
    X = ns["X"]
    X_encoded = ns["X_encoded"]
    y = ns["y"]
    cramers_v = ns["cramers_v"]
    model.eval()

    # ── weights ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), OUT_DIR / "mushroom_model.pth")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ mushroom_model.pth  ({n_params:,} parameters)")

    # ── encoding scheme ───────────────────────────────────────────────────────
    feature_columns = list(X_encoded.columns)
    features = []
    for col in X.columns:
        cats = sorted(df[col].unique())
        dropped = cats[0]  # get_dummies(drop_first=True) drops the first category
        label, icon = FEATURE_LABELS[col]
        vc = df[col].value_counts()
        values = []
        for cat in cats:
            tr = VALUE_LABELS[col].get(cat, cat)
            values.append({
                "code": cat,
                "label": tr,
                "count": int(vc[cat]),
                "swatch": COLOR_SWATCH.get(tr),
                # encoded columns that must be set to 1.0 for this category
                "columns": [] if cat == dropped else [f"{col}_{cat}"],
            })
        features.append({
            "key": col,
            "label": label,
            "icon": icon,
            "dropped": dropped,
            "values": values,
            "columns": [f"{col}_{c}" for c in cats if c != dropped],
        })

    meta = {
        "input_features": len(feature_columns),
        "feature_columns": feature_columns,
        "features": features,
        "dropped_from_dataset": ["veil-type", "gill-attachment"],
        "architecture": str(model),
        "n_params": n_params,
    }
    (OUT_DIR / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(f"✓ model_meta.json     ({len(feature_columns)} encoded columns, "
          f"{len(features)} features)")

    # ── the encoded dataset, exactly as the notebook built it ────────────────
    np.savez_compressed(
        OUT_DIR / "encoded_dataset.npz",
        X=X_encoded.values.astype(np.float32),
        y=y.values.astype(np.int64),
        raw=X.values.astype("U2"),
    )
    print(f"✓ encoded_dataset.npz ({X_encoded.shape[0]} rows)")

    # ── insights ──────────────────────────────────────────────────────────────
    insights = build_insights(ns, features)
    (OUT_DIR / "insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2)
    )
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    (WEB_DATA / "insights.json").write_text(
        json.dumps(insights, ensure_ascii=False, indent=2)
    )
    print("✓ insights.json")
    print("\nExport complete.")


def build_insights(ns: dict, features: list) -> dict:
    df = ns["df"]
    cramers_v = ns["cramers_v"]
    cm = ns["cm"]

    total = len(df)
    edible = int((df["class"] == "e").sum())
    poisonous = int((df["class"] == "p").sum())

    per_feature = []
    breakdown = {}
    for feat in features:
        key = feat["key"]
        v = float(cramers_v(df[key], df["class"]))

        # accuracy of the trivial "majority class per category" rule
        ct = df.groupby([key, "class"]).size().unstack(fill_value=0)
        for cls in ("e", "p"):
            if cls not in ct:
                ct[cls] = 0
        rule_acc = float(ct[["e", "p"]].max(axis=1).sum() / total)

        values = []
        for val in feat["values"]:
            code = val["code"]
            n_e = int(ct.loc[code, "e"])
            n_p = int(ct.loc[code, "p"])
            n = n_e + n_p
            values.append({
                "code": code,
                "label": val["label"],
                "count": n,
                "share": round(n / total * 100, 1),
                "edible": n_e,
                "poisonous": n_p,
                "poison_rate": round(n_p / n * 100, 1) if n else 0.0,
                "swatch": val["swatch"],
            })

        per_feature.append({
            "key": key,
            "label": feat["label"],
            "icon": feat["icon"],
            "cramers_v": round(v, 4),
            "rule_accuracy": round(rule_acc * 100, 2),
            "n_values": len(values),
        })
        breakdown[key] = {
            "label": feat["label"],
            "icon": feat["icon"],
            "values": sorted(values, key=lambda d: -d["count"]),
        }

    per_feature.sort(key=lambda d: -d["cramers_v"])

    # deterministic rules: a single category that always means one class
    rules = []
    for key, data in breakdown.items():
        for val in data["values"]:
            if val["count"] >= 40 and val["poison_rate"] in (0.0, 100.0):
                rules.append({
                    "feature": key,
                    "feature_label": data["label"],
                    "icon": data["icon"],
                    "value": val["label"],
                    "count": val["count"],
                    "share": val["share"],
                    "verdict": "poisonous" if val["poison_rate"] == 100.0 else "edible",
                })
    rules.sort(key=lambda d: -d["count"])

    return {
        "dataset": {
            "total": total,
            "edible": edible,
            "poisonous": poisonous,
            "edible_pct": round(edible / total * 100, 1),
            "poisonous_pct": round(poisonous / total * 100, 1),
            "n_raw_features": int(df.shape[1] - 1),
            "n_model_features": len(features),
            "n_encoded": int(ns["X_encoded"].shape[1]),
        },
        "feature_importance": per_feature,
        "breakdown": breakdown,
        "rules": rules[:14],
        "model": {
            "architecture": "94 → 64 → ReLU → 64 → ReLU → 1 → Sigmoid",
            "epochs": int(ns["EPOCHS"]),
            "batch_size": int(ns["BATCH_SIZE"]),
            "optimizer": "Adam (lr=0.001)",
            "loss": "BCELoss",
            "n_params": sum(p.numel() for p in ns["model"].parameters()),
            "train_accuracy": round(float(ns["train_accuracies"][-1]), 2),
            "test_accuracy": round(float(ns["test_accuracies"][-1]), 2),
            "final_train_loss": float(ns["train_losses"][-1]),
            "final_test_loss": float(ns["test_losses"][-1]),
            "confusion_matrix": [[int(x) for x in row] for row in cm],
            "train_size": int(len(ns["X_train"])),
            "test_size": int(len(ns["X_test"])),
        },
        "curves": {
            "train_loss": [float(x) for x in ns["train_losses"]],
            "test_loss": [float(x) for x in ns["test_losses"]],
            "train_acc": [float(x) for x in ns["train_accuracies"]],
            "test_acc": [float(x) for x in ns["test_accuracies"]],
        },
    }


if __name__ == "__main__":
    main()
