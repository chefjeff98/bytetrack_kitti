import json
import os
import cv2
import pandas as pd
from collections import defaultdict

## NOTES
#
# python convert_labels.py to convert ground truth data to COCO JSON format
# TODO: Add code to generate MOT format labels for evaluation (see Claude)
# To run ByteTrack:
# python tools/track.py -f .\exps\yolox_s_kitti.py -c .\pretrained\yolox_s.pth --fp16 --fuse --test -b 1
#
## END NOTES

# ============ CONFIGURATION ============
# Set these paths for your system
KITTI_ROOT = 'datasets/kitti'
OUTPUT_PATH = 'datasets/kitti/image_02/annotations'

# Choose preprocessing (for pretrained models, use NO preprocessing)
USE_PREPROCESSING = False  # Set to True if you want cropping/resizing
TARGET_IMG_SIZE = (1280, 384) if USE_PREPROCESSING else None
CROP_SIZE = 0 if USE_PREPROCESSING else 0

# Choose which classes to include
INCLUDE_ALL_CLASSES = False  # Set to True to include pedestrians, cyclists, etc.

if INCLUDE_ALL_CLASSES:
    CATEGORIES = {
        'Car': 1, 'Van': 2, 'Truck': 3, 'Tram': 4,
        'Pedestrian': 5, 'Person': 6, 'Cyclist': 7, 'Misc': 8
    }
else:
    #CATEGORIES = {'Car': 2, 'Truck': 7, 'Van': 5, 'Pedestrian': 0}
    CATEGORIES = {'Car': 2}

# ============ CONVERSION FUNCTIONS ============

def get_image_size(img_path):
    """Get image dimensions, accounting for cropping if enabled"""
    img = cv2.imread(img_path)
    
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    
    height, width = img.shape[:2]
    
    if USE_PREPROCESSING:
        height -= CROP_SIZE
        width, height = TARGET_IMG_SIZE
    
    return width, height

def create_mot_ground_truth(json_file, output_base_dir):
    """Convert COCO format JSON to MOT format ground truth txt files"""
    import json
    
    # Load COCO JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Group annotations by video_id
    from collections import defaultdict
    video_annotations = defaultdict(list)
    
    # Create image_id to frame_id and video_id mapping
    image_info = {}
    for img in data['images']:
        image_info[img['id']] = {
            'frame_id': img['frame_id'],
            'video_id': img['video_id']
        }
    
    # Group annotations by video
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id in image_info:
            info = image_info[img_id]
            video_annotations[info['video_id']].append({
                'frame_id': info['frame_id'],
                'track_id': ann['track_id'],
                'bbox': ann['bbox'],
                'category_id': ann['category_id']
            })
    
    # Write MOT format files for each video
    for video_id, annotations in video_annotations.items():
        video_name = f"{video_id - 1:04d}"  # Changed: just "0000" not "kitti-0000"
        
        # Create MOT-style directory structure
        video_dir = os.path.join(output_base_dir, video_name, 'gt')
        os.makedirs(video_dir, exist_ok=True)
        
        output_file = os.path.join(video_dir, 'gt.txt')
        
        # Sort by frame_id
        annotations.sort(key=lambda x: (x['frame_id'], x['track_id']))
        
        with open(output_file, 'w') as f:
            for ann in annotations:
                # MOT format: frame, id, left, top, width, height, conf, x, y, z
                frame = ann['frame_id'] + 1  # MOT uses 1-based frame numbers
                track_id = ann['track_id']
                left, top, width, height = ann['bbox']
                
                # Write in MOT format
                f.write(f"{frame},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")
        
        print(f"Created ground truth: {output_file}")


def convert_bbox(left, top, right, bottom, orig_width, orig_height):
    """Convert KITTI bbox to COCO format with optional preprocessing"""
    
    original_left = left  # SAVE ORIGINAL
    
    if USE_PREPROCESSING:
        # Adjust for cropping
        top = max(0, top - CROP_SIZE)
        bottom = max(0, bottom - CROP_SIZE)
        
        # Scale to new dimensions
        scale_x = TARGET_IMG_SIZE[0] / orig_width
        scale_y = TARGET_IMG_SIZE[1] / (orig_height - CROP_SIZE)
        
        left *= scale_x
        right *= scale_x
        top *= scale_y
        bottom *= scale_y
        
        # ADD THIS DEBUG
        print(f"  BBOX: orig_left={original_left:.1f} -> scaled_left={left:.1f} (scale={scale_x:.4f})")
    
    # Convert to COCO format [x, y, width, height]
    x = left
    y = top
    w = right - left
    h = bottom - top
    
    return [x, y, w, h]

def process_sequence(seq_id, label_path, image_dir):
    """Process one KITTI sequence"""
    # Read label file
    labels_df = pd.read_csv(label_path, delimiter=' ', header=None)
    
    # Column indices: 0=frame, 1=track_id, 2=class, 6-9=bbox
    labels_df = labels_df[[0, 1, 2, 6, 7, 8, 9]]
    labels_df.columns = ['frame', 'track_id', 'class', 'left', 'top', 'right', 'bottom']
    
    images = []
    annotations = []
    annotation_id = 1
    
    # Get all frames in this sequence
    frames = sorted(labels_df['frame'].unique())
    
    # Create mapping of frame to image_id for prev/next links
    frame_to_image_id = {}
    for idx, frame_id in enumerate(frames):
        image_id = int(f"{seq_id:04d}{frame_id:06d}")
        frame_to_image_id[frame_id] = image_id
    
    for idx, frame_id in enumerate(frames):
        # Image info
        img_filename = f"{seq_id:04d}/{frame_id:06d}.png"
        img_path = os.path.join(image_dir, f"{seq_id:04d}", f"{frame_id:06d}.png")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found, skipping: {img_path}")
            continue
        
        try:
            width, height = get_image_size(img_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping frame")
            continue
            
        orig_width, orig_height = cv2.imread(img_path).shape[1::-1]
        
        # Create unique image ID
        image_id = int(f"{seq_id:04d}{frame_id:06d}")
        
        # Determine prev/next image IDs
        prev_image_id = frame_to_image_id[frames[idx - 1]] if idx > 0 else -1
        next_image_id = frame_to_image_id[frames[idx + 1]] if idx < len(frames) - 1 else -1
        
        images.append({
            'id': image_id,
            'file_name': img_filename,
            'width': width,
            'height': height,
            'frame_id': int(frame_id),      # Frame number in sequence
            'prev_image_id': prev_image_id,  # Previous frame ID
            'next_image_id': next_image_id,  # Next frame ID
            'video_id': seq_id + 1               # Sequence ID
        })
        
        # Get all objects in this frame
        frame_labels = labels_df[labels_df['frame'] == frame_id]
        
        for _, obj in frame_labels.iterrows():
            # Filter by class
            class_name = obj['class']
            
            if class_name == 'DontCare':
                continue
            
            if class_name not in CATEGORIES:
                if not INCLUDE_ALL_CLASSES:
                    continue
            
            # Convert bbox
            bbox = convert_bbox(
                obj['left'], obj['top'], obj['right'], obj['bottom'],
                orig_width, orig_height
            )
            
            # Skip invalid boxes
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': CATEGORIES[class_name],
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'track_id': int(obj['track_id'])  # Add track ID from KITTI
            })
            annotation_id += 1
    
    return images, annotations

def convert_kitti_to_coco():
    """Main conversion function"""
    label_dir = os.path.join(KITTI_ROOT, 'label_02')
    image_dir = os.path.join(KITTI_ROOT, 'image_02')
    
    all_images = []
    all_annotations = []
    all_videos = []
    
    # Process each sequence
    for label_file in sorted(os.listdir(label_dir)):
        if not label_file.endswith('.txt'):
            continue
        
        seq_id = int(label_file.replace('.txt', ''))
        label_path = os.path.join(label_dir, label_file)
        
        print(f"Processing sequence {seq_id:04d}...")
        
        images, annotations = process_sequence(seq_id, label_path, image_dir)
        all_images.extend(images)
        all_annotations.extend(annotations)
        
        # Add video entry for this sequence
        all_videos.append({
            'id': seq_id + 1,
            'name': f"{seq_id:04d}"
        })
    
    # Create COCO dataset
    coco_dataset = {
        'info': {
            'description': 'KITTI Tracking Dataset',
            'version': '1.0',
            'year': 2024,
        },
        'videos': all_videos,
        'images': all_images,
        'annotations': all_annotations,
        'categories': [
            {'id': cat_id, 'name': cat_name, 'supercategory': 'object'}
            for cat_name, cat_id in CATEGORIES.items()
        ]
    }
    
    # Save to JSON
    output_file = os.path.join(OUTPUT_PATH, 'kitti_tracking_coco.json')
    with open(output_file, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Total images: {len(all_images)}")
    print(f"Total annotations: {len(all_annotations)}")
    print(f"Output file: {output_file}")

if __name__ == '__main__':
    convert_kitti_to_coco()
    create_mot_ground_truth(
        'datasets/kitti/image_02/annotations/kitti_tracking_coco.json',
        'datasets/mot/train'
    )