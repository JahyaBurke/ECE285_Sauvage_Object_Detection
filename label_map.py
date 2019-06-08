label_to_encoding = { #Labels from Pascal data set
    'bicycle': 1,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'cow': 9,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'sheep': 16,
    'train': 18
    }

coco_to_pascal = { #Maps COCO from Pascal labels
    1:14,
    2:1,
    3:6,
    4:13,
    6:5,
    7:18,
    17:7,
    18:11,
    19:12,
    20:16,
    21:9,
}

pascal_to_coco = {v: k for k,v in coco_to_pascal.items()}

encoding_to_label = {v: k for k,v in label_to_encoding.items()}
