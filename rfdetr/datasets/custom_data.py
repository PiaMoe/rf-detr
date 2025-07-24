import os
import json
from datetime import datetime
from PIL import Image
from tqdm import tqdm

def yolo_to_coco(dataset_dir, subsets=['train', 'val', 'test'], output_dir='annotations', class_names=None):
    os.makedirs(output_dir, exist_ok=True)

    # If no class list is given, just use generic class IDs
    if class_names is None:
        class_names = []

    # Create category section
    categories = [{'id': i, 'name': name} for i, name in enumerate(class_names)]

    info = {
        "description": "",
        "url": "",
        "year": datetime.now().year,
        "contributor": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    licenses = [{
    }]

    for subset in subsets:
        image_id = 0
        annotation_id = 0
        coco_output = {
            'info': info,
            'licenses': licenses,
            'images': [],
            'annotations': [],
            'categories': categories
        }

        label_dir = os.path.join(dataset_dir, subset, 'labels')
        image_dir = os.path.join(dataset_dir, subset, 'images')
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        for label_file in tqdm(label_files, desc=f"Processing {subset}"):
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_file)
            if not os.path.exists(image_path):
                image_file = os.path.splitext(label_file)[0] + '.png'
                image_path = os.path.join(image_dir, image_file)
                if not os.path.exists(image_path):
                    continue  # Skip if no corresponding image

            # Get image size
            with Image.open(image_path) as img:
                width, height = img.size

            # Add image entry
            coco_output['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': width,
                'height': height
            })

            # Read annotations
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 8:
                        continue
                    class_id, x_center, y_center, w, h, dist, cos, sin = map(float, parts)
                    class_id = int(class_id)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    x_min = x_center - w / 2
                    y_min = y_center - h / 2

                    coco_output['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': class_id,
                        'bbox': [x_min, y_min, w, h],
                        'distance': dist,
                        'heading': [cos, sin],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    annotation_id += 1

            image_id += 1

        # Save JSON
        with open(os.path.join(output_dir, f'{subset}.json'), 'w') as f:
            json.dump(coco_output, f, indent=2)

if __name__ == "__main__":
    dataset_dir = '../../../../data/debug_data_Head/'
    output_dir = dataset_dir + '/annotations'
    class_names = ['boat', 'buoy']

    yolo_to_coco(dataset_dir, subsets=['train', 'val', 'test'], output_dir=output_dir, class_names=class_names)