import os
import json
import cv2
import math
from rfdetr import RFDETRBase
import time
from glob import glob
import torch

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def get_class_color_with_distance(cls_name, distance):
    base_colors = {
        'boat': (0, 100, 0),   # BGR
        'buoy': (0, 0, 100),
    }
    base_color = base_colors.get(cls_name, (50, 50, 50))
    factor = 1.0 if distance <= 50 else 1.5 if distance <= 150 else 2.0 if distance <= 300 else 2.5
    return tuple(int(min(c * factor, 255)) for c in base_color)


def annotate_image_cv2(image, detections, classes):
    for det in detections:
        box = list(map(int, det["xyxy"]))  # [x1, y1, x2, y2]
        class_id = det["class_id"]
        class_name = classes[class_id]
        conf = det["confidence"]
        dist = det["distance"]
        head = det["heading"]

        color = get_class_color_with_distance(class_name, dist)
        label = f"{class_name} {conf:.2f} {dist:.1f}m {head:.1f}deg"

        # Box zeichnen
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness=2)

        # Textgröße bestimmen
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Hintergrund für Text
        text_x = box[0]
        text_y = box[1] - 5 if box[1] > 20 else box[1] + text_h + 5
        cv2.rectangle(image, (text_x, text_y - text_h), (text_x + text_w, text_y), color, -1)

        # Text
        text_color = (255, 255, 255) if dist <= 300 else (0, 0, 0)
        cv2.putText(image, label, (text_x, text_y - 2), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Heading-Pfeil
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        rad = math.radians(head)
        box_diag = math.hypot(box[2] - box[0], box[3] - box[1])
        img_diag = math.hypot(image.shape[0], image.shape[1])
        arrow_length = max(0.03 * img_diag, min(0.15 * img_diag, box_diag * 0.5))

        end_x = int(center_x + arrow_length * math.sin(rad))
        end_y = int(center_y + arrow_length * math.cos(rad))
        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), color, thickness=2, tipLength=0.2)

    return image

def inference(run_name, data_path, weights_path, classes):
    output_path = f"../runs/detect/{run_name}"
    os.makedirs(output_path, exist_ok=True)

    model = RFDETRBase(pretrain_weights=weights_path)
    #model.optimize_for_inference()

    if data_path.endswith(".mp4"):  # video mode
        cap = cv2.VideoCapture(data_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(output_path, f"{run_name}_annotated.mp4")
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_start = time_synchronized()
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start = time_synchronized()
            detections = model.predict(frame, threshold=0.25)
            infer_time = time_synchronized() - start
            print(f"Frame {frame_count}/ {total_frames}: inference time = {infer_time:.3f}s")

            if detections:
                image_annotated = annotate_image_cv2(frame, detections, classes=classes)
            else:
                image_annotated = frame  # unbearbeitet

            out_video.write(image_annotated)
            frame_count += 1

        cap.release()
        out_video.release()
        total_time = time_synchronized() - total_start
        print(f"\ninference time total video: {total_time:.2f}s")


    else:
        output_img_path = os.path.join(output_path, "images")
        output_label_path = os.path.join(output_path, "labels")
        os.makedirs(output_img_path, exist_ok=True)
        os.makedirs(output_label_path, exist_ok=True)
        image_paths = sorted(glob(os.path.join(data_path, "*.jpg")) + glob(os.path.join(data_path, "*.png")))

        for i, path in enumerate(image_paths):
            img_name = os.path.basename(path)
            label_name = img_name.rsplit(".", 1)[0] + ".json"
            save_label = os.path.join(output_label_path, label_name)
            save_image = os.path.join(output_img_path, img_name)
            image = cv2.imread(path)
            start = time_synchronized()
            detections = model.predict(image, threshold=0.25)
            infer_time = time_synchronized() - start
            print(f"image {i + 1}/{len(image_paths)} ({img_name}): inference time = {infer_time:.3f}s")

            if not detections:
                continue

            image_annotated = annotate_image_cv2(image, detections, classes)
            cv2.imwrite(save_image, image_annotated)

            detection_data = [
                {
                    "class_id": det["class_id"],
                    "class_name": classes[det["class_id"]],
                    "confidence": det["confidence"],
                    "bbox": list(map(float, det["xyxy"])),
                    "distance": det["distance"],
                    "heading": det["heading"]
                }
                for det in detections
            ]
            with open(save_label, "w") as f:
                json.dump(detection_data, f, indent=2)


if __name__ == "__main__":
    run_name = "DetDistHeadFreezeMLP"
    data_path = "../../../data/BOArDING_Dataset/BOArDING_Det/val/images"
    weights_path = "../runs/train/DetDistHead_Freeze/checkpoint_best_total.pth"
    classes = ['boat', 'buoy']
    inference(run_name, data_path, weights_path, classes)
