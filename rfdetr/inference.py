import io
import os
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.custom_classes import CLASSES
import json


run_name = "val"
data_path = "../../../data/BOArDING_Dataset/BOArDING_Det/for_detr"
output_path = f"../runs/detect/{run_name}"
output_img_path = os.path.join(output_path, "images")
os.makedirs(output_img_path, exist_ok=True)
output_label_path = os.path.join(output_path, "labels")
os.makedirs(output_label_path, exist_ok=True)


ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{data_path}/val",
    annotations_path=f"{data_path}/val/val.json",
)

model = RFDETRBase(pretrain_weights="../runs/train/test_run/checkpoint_best_total.pth")
#model = model.optimize_for_inference()


for i, data in enumerate(ds):
    path, image_content, annotations = data
    img_name = os.path.basename(path)
    label_name = img_name.replace(".jpg", ".json")
    save_label = os.path.join(output_label_path, label_name)

    image = Image.open(path)
    detections = model.predict(image, threshold=0.25)

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(int(image.size[0]/2), int(image.size[1]/2)))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True)


    detections_labels = [
        f"{ds.classes[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, detections_labels)
    save_image = os.path.join(output_img_path, img_name)
    annotated_image.save(save_image)

    detection_data = [
        {
            "class_id": int(class_id),
            "class_name": ds.classes[class_id],
            "confidence": float(conf),
            "bbox": list(map(float, box))  # [x1, y1, x2, y2]
        }
        for class_id, conf, box in zip(
            detections.class_id,
            detections.confidence,
            detections.xyxy
        )
    ]

    with open(save_label, "w") as f:
        json.dump(detection_data, f, indent=2)




