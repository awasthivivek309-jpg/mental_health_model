import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import sounddevice as sd


def pad_or_trim(y: np.ndarray, target_length: int) -> np.ndarray:
    if len(y) > target_length:
        return y[:target_length]
    if len(y) < target_length:
        return np.pad(y, (0, target_length - len(y)))
    return y


def preprocess_audio_array(
    y_raw: np.ndarray,
    sr_raw: int,
    target_sr: int,
    target_duration: float,
    normalize: bool = True,
) -> tuple[np.ndarray, dict]:
    duration_raw = len(y_raw) / float(sr_raw) if sr_raw else 0.0
    warning_msgs = []

    if sr_raw != target_sr:
        warning_msgs.append(f"SR mismatch: {sr_raw} -> {target_sr}")
    if abs(duration_raw - target_duration) > 0.05:
        warning_msgs.append(f"Duration mismatch: {duration_raw:.2f}s -> {target_duration:.2f}s")

    y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=target_sr) if sr_raw != target_sr else y_raw
    target_samples = int(target_sr * target_duration)
    y = pad_or_trim(y, target_samples)

    if normalize:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak

    meta = {
        "original_sr": sr_raw,
        "original_duration": duration_raw,
        "target_sr": target_sr,
        "target_duration": target_duration,
        "num_samples": len(y),
        "warnings": warning_msgs,
    }
    return y.astype(np.float32), meta


def extract_feature_vector(y: np.ndarray, sr: int, n_mfcc: int = 40, n_mels: int = 128) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    features = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
            np.mean(mel_db, axis=1),
            np.std(mel_db, axis=1),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_model_bundle(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    required_keys = {"model", "scaler", "label_encoder", "target_sr", "target_duration"}
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def listen_once(seconds: float, sample_rate: int) -> np.ndarray:
    print(f"Listening for {seconds:.1f}s...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def predict_emotion_from_mic(bundle: dict, listen_seconds: float | None = None) -> dict:
    model = bundle["model"]
    scaler = bundle["scaler"]
    label_encoder = bundle["label_encoder"]
    target_sr = int(bundle["target_sr"])
    target_duration = float(bundle["target_duration"])

    seconds = float(listen_seconds) if listen_seconds is not None else target_duration
    y_mic = listen_once(seconds=seconds, sample_rate=target_sr)

    y_proc, meta = preprocess_audio_array(
        y_raw=y_mic,
        sr_raw=target_sr,
        target_sr=target_sr,
        target_duration=target_duration,
        normalize=True,
    )

    feature = extract_feature_vector(y_proc, sr=target_sr).reshape(1, -1)
    feature_scaled = scaler.transform(feature)

    probs = model.predict_proba(feature_scaled)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = str(label_encoder.inverse_transform([pred_idx])[0])

    prob_dict = {
        str(cls): float(prob)
        for cls, prob in zip(label_encoder.classes_, probs)
    }

    return {
        "predicted_emotion": pred_label,
        "probabilities": prob_dict,
        "preprocess_meta": meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen from microphone and predict speech emotion.")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ser_mlp_pipeline.joblib",
        help="Path to exported model bundle (.joblib or .pkl).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one prediction and exit.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Override listening duration in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    bundle = load_model_bundle(model_path)

    print("Model loaded successfully.")
    print(f"Model path: {model_path.resolve()}")
    print(f"Classes: {[str(c) for c in bundle['label_encoder'].classes_]}")

    if args.once:
        result = predict_emotion_from_mic(bundle, listen_seconds=args.seconds)
        print("Prediction result:")
        print(result)
        return

    print("Press ENTER to listen, or type q then ENTER to quit.")
    while True:
        cmd = input("Command: ").strip().lower()
        if cmd == "q":
            print("Exiting.")
            break

        result = predict_emotion_from_mic(bundle, listen_seconds=args.seconds)
        top = result["predicted_emotion"]
        conf = max(result["probabilities"].values())

        print(f"Predicted emotion: {top} (confidence={conf:.4f})")
        print("Probability scores:")
        for label, score in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {label:>8}: {score:.4f}")

        warnings_list = result["preprocess_meta"].get("warnings", [])
        if warnings_list:
            print("Preprocessing warnings:")
            for w in warnings_list:
                print(f"  - {w}")


if __name__ == "__main__":
    main()
