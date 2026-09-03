import os
import sys
import json
import requests
from ultralytics import YOLOWorld

IMAGE_URL = sys.argv[1] if len(sys.argv) > 1 else None
TASK_ID = sys.argv[2] if len(sys.argv) > 2 else "latest"

def run():
    if not IMAGE_URL:
        return

    # Скачиваем входное фото
    resp = requests.get(IMAGE_URL, timeout=20)
    with open("input.jpg", "wb") as f:
        f.write(resp.content)

    # Запускаем YOLO-World
    model = YOLOWorld("yolov8s-worldv2.pt")
    target_classes = [
        "sheet of paper", "signboard", "holding paper", 
        "placard", "notebook", "white banner", "sketchbook"
    ]
    model.set_classes(target_classes)

    results = model.predict("input.jpg", conf=0.15)
    boxes = results[0].boxes

    # Собираем чистый JSON с координатами
    detected_boxes = []
    for box in boxes:
        coords = box.xywh[0].cpu().numpy().astype(int).tolist() # [x_center, y_center, width, height]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = target_classes[cls_id] if cls_id < len(target_classes) else "paper"
        detected_boxes.append({
            "label": label,
            "conf": round(conf, 2),
            "x": coords[0],
            "y": coords[1],
            "w": coords[2],
            "h": coords[3]
        })

    output_data = {
        "task_id": TASK_ID,
        "count": len(detected_boxes),
        "boxes": detected_boxes
    }

    # Сохраняем в папку results/
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{TASK_ID}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Результаты сохранены в {out_path}: найдено {len(detected_boxes)} зон")

if __name__ == "__main__":
    run()
