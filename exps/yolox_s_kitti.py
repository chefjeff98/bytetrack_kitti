# encoding: utf-8
import os
import torch
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model configuration - MUST match your pretrained model
        self.num_classes = 80  # COCO has 80 classes (pretrained YOLOX uses this)
        self.depth = 0.33      # For YOLOX-S
        self.width = 0.50      # For YOLOX-S
        
        # If using YOLOX-M, change to:
        # self.depth = 0.67
        # self.width = 0.75
        
        # If using YOLOX-X, change to:
        # self.depth = 1.33
        # self.width = 1.25
        
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Image size - KITTI original is 1242x375
        # Keep original size for pretrained models
        self.test_size = (384, 1280)   # <-- USE THIS (matches KITTI AR, no downsampling)
        #self.test_size = (608, 1088)  # (height, width) - common for MOT
        # Or use KITTI size: self.test_size = (384, 1248)  # Closest to 375x1242
        
        # Detection thresholds
        self.test_conf = 0.001   # Confidence threshold (lower = more detections)
        self.nmsthre = 0.7      # NMS threshold
        
        # Tracking parameters (for ByteTrack algorithm)
        #self.track_thresh = 0.3  # confidence threshold FIXME: do I need this?
        self.track_buffer = 30   # Frames to keep lost tracks
        self.match_thresh = 0.8  # IOU threshold for matching
        
        # Dataset paths - UPDATE THESE
        self.val_ann = "kitti_tracking_coco.json"  # Your converted COCO JSON
        
    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform
        
        valdataset = MOTDataset(
            data_dir=os.path.join(self.get_data_dir(), "kitti", "image_02"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='',  # Empty for custom dataset
            preproc=ValTransform(
                rgb_means=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            ),
        )

        # Support all 80 COCO classes
        valdataset.class_ids = list(range(80))
        
        if is_distributed:
            batch_size = batch_size // torch.distributed.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {
            "num_workers": 4,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        
        return val_loader
    
    def get_data_dir(self):
        """Override to point to your KITTI location"""
        # UPDATE THIS PATH
        return "datasets"  # Folder containing 'kitti' subfolder
    
    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator
        
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator