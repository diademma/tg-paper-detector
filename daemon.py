import os
import sys
import time
import json
import cv2
import numpy as np
import subprocess
from ultralytics import YOLOWorld

MAX_LIFETIME = 18000 
POLL_INTERVAL = 2

def git_cmd(cmd_list):
    try:
        subprocess.run(cmd_list, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def is_bright_paper(img, x1, y1, x2, y2):
    """Железная проверка: бумага ДОЛЖНА быть светлой (отсекает черные худи и волосы)"""
    try:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        
        # Переводим в градации серого и считаем среднюю яркость
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Если средняя яркость меньше 125 — это темная одежда/волосы, а не бумага!
        return mean_brightness >= 125
    except Exception:
        return True

def main():
    print("🧠 Загрузка YOLOv8x-World...")
    model = YOLOWorld("yolov8x-worldv2.pt")

    # Идеальный набор классов: от белых листов до пустых табличек в руках
    target_classes = [
        "white paper", 
        "sheet of paper", 
        "blank paper", 
        "blank sign", 
        "white board",
        "held sign",
        "notepad", 
        "sketchbook"
    ]
    model.set_classes(target_classes)
    print("⚡ ДЕМОН АКТИВЕН (С умным фильтром белой бумаги)!")

    os.makedirs("tasks", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    start_time = time.time()

    while time.time() - start_time < MAX_LIFETIME:
        git_cmd(["git", "pull", "--rebase"])

        task_files = [f for f in os.listdir("tasks") if f.endswith(".jpg") or f.endswith(".png")]

        if task_files:
            for filename in task_files:
                task_id = os.path.splitext(filename)[0]
                img_path = os.path.join("tasks", filename)
                print(f"🎯 Анализ: {task_id}")

                try:
                    # Загружаем изображение через OpenCV для проверки пикселей
                    orig_cv_img = cv2.imread(img_path)
                    img_h, img_w = orig_cv_img.shape[:2]

                    # Порог 0.05 захватит абсолютно все таблички
                    results = model.predict(img_path, conf=0.05, imgsz=1280)
                    boxes = results[0].boxes

                    detected_boxes = []
                    for box in boxes:
                        coords = box.xywh[0].cpu().numpy().astype(int).tolist()
                        cx, cy, bw, bh = coords[0], coords[1], coords[2], coords[3]
                        
                        # Координаты углов [x1, y1, x2, y2]
                        x1 = max(0, int(cx - bw / 2))
                        y1 = max(0, int(cy - bh / 2))
                        x2 = min(img_w, int(cx + bw / 2))
                        y2 = min(img_h, int(cy + bh / 2))

                        # 1. Отсекаем рамки во весь экран (> 90% всей картинки)
                        if bw > img_w * 0.92 and bh > img_h * 0.92:
                            continue

                        # 2. АНТИ-ХУДИ ФИЛЬТР: проверяем, что найденная зона реально белая/светлая
                        if not is_bright_paper(orig_cv_img, x1, y1, x2, y2):
                            print(f"🛡️ Отсекли темный объект (худи/волосы) на x={cx}, y={cy}")
                            continue

                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = target_classes[cls_id] if cls_id < len(target_classes) else "paper"
                        
                        detected_boxes.append({
                            "label": label,
                            "conf": round(conf, 2),
                            "x": cx,
                            "y": cy,
                            "w": bw,
                            "h": bh
                        })

                    out_data = {
                        "task_id": task_id,
                        "count": len(detected_boxes),
                        "boxes": detected_boxes
                    }

                    with open(f"results/{task_id}.json", "w", encoding="utf-8") as f:
                        json.dump(out_data, f, indent=2)

                    if os.path.exists(img_path):
                        os.remove(img_path)

                    print(f"✅ Готово: {task_id} (найдено {len(detected_boxes)} чистых зон)")

                except Exception as e:
                    print(f"❌ Ошибка: {e}")
                    if os.path.exists(img_path):
                        os.remove(img_path)

            git_cmd(["git", "add", "results/", "tasks/"])
            git_cmd(["git", "commit", "-m", "Processed batch"])
            git_cmd(["git", "push"])

        time.sleep(POLL_INTERVAL)

    print("🔄 Перезапуск 5-часового цикла...")

if __name__ == "__main__":
    main()
