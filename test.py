import glob
import os
from pathlib import Path
from collections import OrderedDict

# Check ground truth files
gt_path = 'datasets/mot/train'
gt_type = ''  # or '_val_half' if you use that

print("="*60)
print("GROUND TRUTH FILES CHECK")
print("="*60)

gtfiles = glob.glob(os.path.join(gt_path, '*/gt/gt{}.txt'.format(gt_type)))
print(f"\nSearching in: {os.path.join(gt_path, '*/gt/gt{}.txt'.format(gt_type))}")
print(f"Found {len(gtfiles)} ground truth files:")
for f in gtfiles:
    print(f"  - {f}")

if gtfiles:
    print("\nExtracted sequence names from ground truth:")
    gt = OrderedDict([(Path(f).parts[-3], f) for f in gtfiles])
    for seq_name, filepath in gt.items():
        print(f"  - Sequence: '{seq_name}' -> {filepath}")

# Check tracking results
results_folder = './YOLOX_outputs/yolox_s_kitti/track_results_sort'
print("\n" + "="*60)
print("TRACKING OUTPUT FILES CHECK")
print("="*60)

tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) 
           if not os.path.basename(f).startswith('eval')]
print(f"\nSearching in: {results_folder}")
print(f"Found {len(tsfiles)} tracking result files:")
for f in tsfiles:
    print(f"  - {f}")

if tsfiles:
    print("\nExtracted sequence names from tracking results:")
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], f) for f in tsfiles])
    for seq_name, filepath in ts.items():
        print(f"  - Sequence: '{seq_name}' -> {filepath}")

# Check matching
if gtfiles and tsfiles:
    print("\n" + "="*60)
    print("MATCHING CHECK")
    print("="*60)
    
    gt_names = set(Path(f).parts[-3] for f in gtfiles)
    ts_names = set(os.path.splitext(Path(f).parts[-1])[0] for f in tsfiles)
    
    print(f"\nGround truth sequences: {sorted(gt_names)}")
    print(f"Tracking result sequences: {sorted(ts_names)}")
    
    matched = gt_names & ts_names
    only_in_gt = gt_names - ts_names
    only_in_ts = ts_names - gt_names
    
    print(f"\nMatched sequences ({len(matched)}): {sorted(matched)}")
    if only_in_gt:
        print(f"Only in ground truth ({len(only_in_gt)}): {sorted(only_in_gt)}")
    if only_in_ts:
        print(f"Only in tracking results ({len(only_in_ts)}): {sorted(only_in_ts)}")
    
    if matched:
        print("\n✓ GOOD: Sequences are matching!")
    else:
        print("\n✗ ERROR: No matching sequences found!")
        print("\nPossible issues:")
        print("1. Sequence naming mismatch")
        print("2. Ground truth files in wrong location")
        print("3. Tracking results not generated yet")

print("\n" + "="*60)