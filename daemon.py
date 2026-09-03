import os
import sys
import time
import json
import subprocess
from ultralytics import YOLOWorld

MAX_LIFETIME = 18000 
POLL_INTERVAL = 2

def git_cmd(cmd_list):
    try:
        subprocess.run(cmd_list, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def main():
    print("🧠 Прогрев флагмана: загружаю YOLOv8x-World в память...")
    model = YOLOWorld("yolov8x-worldv2.pt")

    # Исключили 'poster' и 'board', чтобы он не обводил самих персонажей как плакаты
    target_classes = [
        "sheet of paper", 
        "signboard", 
        "placard", 
        "notepad", 
        "sketchbook", 
        "handheld sign"
    ]
    model.set_classes(target_classes)
    print("🚀 ФЛАГМАН СФОКУСИРОВАН НА ПРЕДМЕТАХ В РУКАХ (conf=0.15)!")

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
                print(f"🎯 Анализ задачи: {task_id}")

                try:
                    # Порог 0.15 идеально отсекает ложные срабатывания на фон и лица
                    results = model.predict(img_path, conf=0.15, imgsz=1280)
                    boxes = results[0].boxes

                    detected_boxes = []
                    for box in boxes:
                        coords = box.xywh[0].cpu().numpy().astype(int).tolist()
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

                    out_data = {
                        "task_id": task_id,
                        "count": len(detected_boxes),
                        "boxes": detected_boxes
                    }

                    with open(f"results/{task_id}.json", "w", encoding="utf-8") as f:
                        json.dump(out_data, f, indent=2)

                    if os.path.exists(img_path):
                        os.remove(img_path)

                    print(f"✅ Задача {task_id} готова! Найдено зон: {len(detected_boxes)}")

                except Exception as e:
                    print(f"❌ Ошибка: {e}")
                    if os.path.exists(img_path):
                        os.remove(img_path)

            git_cmd(["git", "add", "results/", "tasks/"])
            git_cmd(["git", "commit", "-m", "Processed batch"])
            git_cmd(["git", "push"])

        time.sleep(POLL_INTERVAL)

    print("🔄 Завершение планового 5-часового цикла...")

if __name__ == "__main__":
    main()
