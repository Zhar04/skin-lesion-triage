"""
Этап 7. Демо-приложение (Gradio): фото + анкета -> уровень риска + Grad-CAM.

Запуск (локально, интернет не нужен, работает на CPU):
  python app/app.py
  python app/app.py --run fusion_efficientnet_b0__seed42 --image-run image_efficientnet_b0__seed42
Откроется http://127.0.0.1:7860

Примеры для демо: положите 4–5 фото из TEST в папку app/examples/ (jpg).
Загруженные фото никуда не сохраняются приложением.
"""
import argparse
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import OUT_DIR  # noqa: E402
from gradcam import compute_cam  # noqa: E402
from models import load_run  # noqa: E402

REGIONS = ["FACE", "NOSE", "EAR", "LIP", "SCALP", "NECK", "CHEST", "BACK", "ABDOMEN",
           "ARM", "FOREARM", "HAND", "THIGH", "FOOT"]
REGION_RU = {"FACE": "Лицо", "NOSE": "Нос", "EAR": "Ухо", "LIP": "Губа", "SCALP": "Волосистая часть головы",
             "NECK": "Шея", "CHEST": "Грудь", "BACK": "Спина", "ABDOMEN": "Живот", "ARM": "Плечо",
             "FOREARM": "Предплечье", "HAND": "Кисть", "THIGH": "Бедро", "FOOT": "Стопа"}
SYMPTOMS = {"itch": "Зудит?", "grew": "Увеличилось в размере?", "hurt": "Болит?",
            "changed": "Изменилось (цвет/форма)?", "bleed": "Кровоточит?", "elevation": "Возвышается над кожей?"}
ANSWER = {"Да": 1.0, "Нет": 0.0, "Не знаю": np.nan}  # "Не знаю" -> пропуск -> заполнится как в train

DISCLAIMER = ("⚠️ **Учебный прототип, не медицинское изделие.** Модель не ставит диагноз и может ошибаться, "
              "особенно для тёмных типов кожи (их почти нет в обучающих данных). "
              "При любых сомнениях обратитесь к дерматологу.")


def load(run, out_dir):
    if run is None or not (out_dir / "models" / run).exists():
        return None
    model, cfg, prep = load_run(out_dir / "models" / run)
    thr_all = json.loads((out_dir / "models" / "thresholds.json").read_text())
    thr = thr_all.get(cfg["run"], {"low": 0.5, "high": 0.8})
    return {"model": model, "cfg": cfg, "prep": prep, "thr": thr}


def risk_text(prob, thr):
    if prob >= thr["high"]:
        return "🔴 **Высокий риск** — рекомендуется осмотр дерматолога в ближайшее время"
    if prob >= thr["low"]:
        return "🟡 **Средний риск** — рекомендуется плановый осмотр дерматолога"
    return ("🟢 **Низкий риск** по оценке модели — но если поражение меняется, кровоточит "
            "или беспокоит, всё равно покажите его врачу")


def build_app(main, image_only):
    def predict(img, age, region, *answers):
        if img is None:
            return "Загрузите фото", None
        row = pd.DataFrame([{"age": age, "region": region,
                             **{k: ANSWER[a] for k, a in zip(SYMPTOMS, answers)}}])
        prob, overlay = compute_cam(main["model"], main["cfg"], main["prep"], img, row)
        text = f"### {risk_text(prob, main['thr'])}\n\nВероятность злокачественного поражения: **{prob:.0%}**"
        if image_only:
            p_img, _ = compute_cam(image_only["model"], image_only["cfg"], None, img)
            text += f"\n\nДля сравнения, модель только по фото: {p_img:.0%}"
        text += (f"\n\n<small>Пороги: низкий/средний {main['thr']['low']:.2f} (ловит ≥90% раков на валидации), "
                 f"средний/высокий {main['thr']['high']:.2f}</small>")
        return text, overlay

    with gr.Blocks(title="Skin Lesion Triage", analytics_enabled=False) as demo:
        gr.Markdown("# 🩺 Skin Lesion Triage\nПредварительная оценка риска кожного поражения по фото "
                    "со смартфона и короткой анкете (PAD-UFES-20, мультимодальная модель).")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Фото поражения (крупно, при хорошем свете)")
                age = gr.Slider(5, 100, value=60, step=1, label="Возраст")
                region = gr.Dropdown([(REGION_RU[r], r) for r in REGIONS], value="FACE", label="Где находится")
                answers = [gr.Radio(list(ANSWER), value="Нет", label=q) for q in SYMPTOMS.values()]
                btn = gr.Button("Оценить риск", variant="primary")
            with gr.Column():
                result = gr.Markdown()
                cam = gr.Image(label="Grad-CAM: куда смотрела модель (красное = важнее)")
        btn.click(predict, [img, age, region, *answers], [result, cam])
        examples = sorted((ROOT / "app" / "examples").glob("*.jpg"))
        if examples:
            gr.Examples([[str(e)] for e in examples], inputs=[img], label="Примеры из тестовой выборки")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fusion_efficientnet_b0__seed42")
    ap.add_argument("--image-run", default="image_efficientnet_b0__seed42")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    out = Path(args.out)
    main_model = load(args.run, out)
    if main_model is None:
        raise SystemExit(f"Не найдена модель outputs/models/{args.run}. Скачайте её из Google Drive (см. ROADMAP, этап 7).")
    build_app(main_model, load(args.image_run, out)).launch(server_port=args.port)


if __name__ == "__main__":
    main()
