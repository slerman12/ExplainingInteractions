import PIL
import torchvision
from cv2 import imread, imshow
import matplotlib.pyplot as plt
from itertools import combinations
from random import shuffle
import torch
import torch.utils.data as data
import os
from PIL import Image
from matplotlib.figure import Figure
from pycocotools.coco import COCO
from skimage import io
import numpy as np
from skimage.viewer.utils import canvas, FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def find_best(interaction_size=2, num_classes=2, num_allowed_intersections=1):
    """Finds the best objects to interact."""
    coco = COCO("./annotations/instances_val2017.json")
    ids = list(coco.imgs.keys())

    best_len = 0
    best_classes = None
    all_cat_ids = [coco_annotations["category_id"] for id in ids for ann_ids in coco.getAnnIds(imgIds=id) for
                   coco_annotations in coco.loadAnns(ann_ids)]
    all_cat_inter_pairs = list(combinations(list(combinations(all_cat_ids, interaction_size)), num_classes))
    for classes in all_cat_inter_pairs:
        if any([len(set(c[0]).intersection(c[1])) > num_allowed_intersections for c in combinations(classes, 2)]):
            continue
        classes = [sorted(c) for c in classes]
        all_classes = [c for _class in classes for c in _class]
        all_positive_cats = []
        all_negative_cats = []
        partial_negative_cats = [[] for _ in classes]
        partial_positive_cats = [[] for _ in classes]
        for i, id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=id)
            coco_annotations = coco.loadAnns(ann_ids)
            categories = [ann["category_id"] for ann in coco_annotations]
            unique_positive_categories = set([cat for cat in categories if cat in all_classes])
            if all([set(c) <= set(categories) for c in classes]):
                all_positive_cats.append(id)
            elif any([set(c) <= set(categories) for c in classes]):
                for c, cl in enumerate(classes):
                    if set(cl) <= set(categories):
                        partial_positive_cats[c].append(id)
            # elif len(unique_positive_categories) > 0:
            elif any([set(pair) <= set(all_classes) for pair in combinations(unique_positive_categories, 2)]):
                p = 0
                for pair in combinations(unique_positive_categories, 2):
                    if set(pair) <= set(all_classes):
                        partial_negative_cats[p].append(id)
                        p += 1
            # else:
            elif len(unique_positive_categories) > 0:
                all_negative_cats.append(id)
        dataset_fourth_len = min(len(all_positive_cats) // 2, len(partial_positive_cats[0]),
                                 len(partial_positive_cats[1]), len(partial_negative_cats[0]),
                                 len(partial_negative_cats[1]), len(all_negative_cats) // 2)
        if dataset_fourth_len > best_len:
            best_len = dataset_fourth_len
            best_classes = (classes, [[coco.loadCats(_c)[0]["name"] for _c in c] for c in classes], dataset_fourth_len)
            print(best_classes)

    return best_classes


# best = find_best()


def find_best2():
    """Finds the best objects to interact."""
    coco = COCO("./annotations/instances_val2017.json")
    ids = list(coco.imgs.keys())
    best_d_len = 0
    best_classes = None
    all_cat_ids = [coco_annotations["category_id"] for id in ids for ann_ids in coco.getAnnIds(imgIds=id) for
                   coco_annotations in coco.loadAnns(ann_ids)]
    for classes in combinations(set(all_cat_ids), 3):
        all_c = 0
        pairs = [0 for _ in range(3)]
        # For each image
        for i, id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=id)
            coco_annotations = coco.loadAnns(ann_ids)
            categories = [ann["category_id"] for ann in coco_annotations]
            # If all class tuples in categories
            if set(classes) <= set(categories):
                all_c += 1
            # If just some class (or not class) tuples in categories
            elif len(set(classes).intersection(categories)) == 2:
                for ind, pair in enumerate(list(combinations(classes, 2))):
                    if set(pair) <= set(categories):
                        pairs[ind] = pairs[ind] + 1
        d_len = [min([pairs[p_] if p != p else pairs[p_] // 3 for p_, _ in enumerate(pairs)])
                 for p, _ in enumerate(pairs)]
        if max(d_len) >= best_d_len:
            best_d_len = max(d_len)
            best_classes = (classes, [coco.loadCats(c)[0]["name"] for c in classes], d_len, all_c, pairs)
            print(best_classes)

    return best_classes


# find_best2()


def find_best3():
    """Finds the best objects to interact."""
    coco = COCO("./annotations/instances_val2017.json")
    ids = list(coco.imgs.keys())
    best_d_len = 0
    best_classes = None
    all_cat_ids = [coco_annotations["category_id"] for id in ids for ann_ids in coco.getAnnIds(imgIds=id) for
                   coco_annotations in coco.loadAnns(ann_ids)]
    for classes in combinations(set(all_cat_ids), 2):
        all_c = 0
        pairs = [0 for _ in range(2)]
        # For each image
        for i, id in enumerate(ids):
            ann_ids = coco.getAnnIds(imgIds=id)
            coco_annotations = coco.loadAnns(ann_ids)
            categories = [ann["category_id"] for ann in coco_annotations]
            # If all class tuples in categories
            if set(classes) <= set(categories):
                all_c += 1
            # If just some class (or not class) tuples in categories
            elif len(set(classes).intersection(categories)) == 1:
                for ind, pair in enumerate(classes):
                    if pair in categories:
                        pairs[ind] = pairs[ind] + 1
        d_len = min([all_c // 2, pairs[0], pairs[1]])
        if d_len >= best_d_len:
            best_d_len = d_len
            best_classes = (classes, [coco.loadCats(c)[0]["name"] for c in classes], d_len, all_c, pairs)
            print(best_classes)

    return best_classes


# find_best3()


# class CocoDataset(data.Dataset):
#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
#
#     def __init__(self, root, json, transform=None):
#         """Set the path for images.
#
#         Args:
#             root: image directory.
#             json: coco annotation file path.
#             transform: image transformer.
#         """
#         self.root = root
#         self.coco = COCO(json)
#         self.transform = transform
#         self.ids = []
#         ids = list(self.coco.imgs.keys())
#
#         # # # TOP:
#         # ((1, 2, 3), ['person', 'bicycle', 'car'], [2, 2, 2], 45, [77, 314, 7])
#         # ((1, 3, 4), ['person', 'car', 'motorcycle'], [2, 2, 2], 51, [308, 77, 8])
#         # ((1, 3, 5), ['person', 'car', 'airplane'], [2, 2, 2], 3, [356, 28, 6])
#         # ((1, 3, 6), ['person', 'car', 'bus'], [6, 6, 6], 82, [277, 61, 20])
#         # ((1, 3, 8), ['person', 'car', 'truck'], [17, 17, 17], 101, [258, 58, 51])
#         # ((1, 44, 47), ['person', 'bottle', 'cup'], [18, 18, 18], 74, [133, 143, 54])
#         # ((1, 47, 51), ['person', 'cup', 'bowl'], [19, 19, 19], 60, [157, 72, 58])
#         # ((1, 47, 67), ['person', 'cup', 'dining table'], [24, 24, 24], 120, [97, 130, 74])
#         # ((1, 62, 67), ['person', 'chair', 'dining table'], [29, 29, 29], 152, [215, 98, 89])
#         # # # # #
#         # ([[1, 3], [6, 8]], [['person', 'car'], ['bus', 'truck']], 27)
#         # self.classes = [(1, 3), (6, 8)]
#         # ([[1, 47], [62, 67]], [['person', 'cup'], ['chair', 'dining table']], 74)
#         # self.classes = [(1, 47), (62, 67)]
#         # ((4, 1, 3), ['motorcycle', 'person', 'car'], 217, 1160, [1626, 217, 7359])
#         self.classes = [(4, 1), (1, 3)]
#         # self.classes = [(1, 62), (62, 67)]
#         self.classes = [sorted(c) for c in self.classes]
#         self.class_names = [[self.coco.loadCats(_c)[0]["name"] for _c in c] for c in self.classes]
#         print([[self.coco.loadCats(_c)[0]["name"] for _c in c] for c in self.classes])
#         all_classes = set([c for _class in self.classes for c in _class])
#         non_classes = [comb for comb in list(combinations(all_classes, 2)) if sorted(comb) not in self.classes]
#
#         all_positive_cats = []
#         partial_positive_cats = [[] for _ in self.classes]
#         partial_negative_cats = [[] for _ in non_classes]
#         for i, id in enumerate(ids):
#             ann_ids = self.coco.getAnnIds(imgIds=id)
#             coco_annotations = self.coco.loadAnns(ann_ids)
#             categories = [ann["category_id"] for ann in coco_annotations]
#             if all_classes <= set(categories):
#                 all_positive_cats.append(id)
#             elif len(all_classes.intersection(categories)) == 2:
#                 for ind, c in enumerate(self.classes):
#                     if set(c) <= set(categories):
#                         partial_positive_cats[ind].append(id)
#                 for ind, c in enumerate(non_classes):
#                     if set(c) <= set(categories):
#                         partial_negative_cats[ind].append(id)
#         frac = (len(partial_positive_cats) + 1) / len(partial_negative_cats)
#         dataset_frac_len = int(min([min([len(p) for p in partial_positive_cats]),
#                                min([len(p) for p in partial_negative_cats]) // frac]))
#         shuffle(all_positive_cats)
#         for p in partial_positive_cats:
#             shuffle(p)
#         for p in partial_negative_cats:
#             shuffle(p)
#         all_positive_cats = all_positive_cats[:min([dataset_frac_len, len(all_positive_cats)])]
#         for p in range(len(partial_positive_cats)):
#             partial_positive_cats[p] = partial_positive_cats[p][:dataset_frac_len]
#         for p in range(len(partial_negative_cats)):
#             partial_negative_cats[p] = partial_negative_cats[p][:int(dataset_frac_len * frac)]
#         print(len(all_positive_cats), len(partial_positive_cats[0]), len(partial_positive_cats[1]),
#               len(partial_negative_cats[0]))
#
#         self.ids = all_positive_cats + partial_positive_cats[0] + partial_positive_cats[1] + partial_negative_cats[0]
#         self.data_len = len(self.ids)
#
#     def __getitem__(self, index):
#         """Returns one data pair (image and class labels)."""
#         ann_ids = self.coco.getAnnIds(imgIds=self.ids[index])
#         coco_annotations = self.coco.loadAnns(ann_ids)
#         cats = [ann["category_id"] for ann in coco_annotations]
#         path = self.coco.loadImgs(self.ids[index])[0]['file_name']
#         img = Image.open(os.path.join(self.root, path)).convert("RGB")
#
#         labels = torch.zeros(len(self.classes))
#         for i, c in enumerate(self.classes):
#             if set(c) <= set(cats):
#                 labels[i] = 1
#         if labels.sum() == len(self.classes):
#             lbl = 0
#         elif labels[0] == 1:
#             lbl = 0
#         elif labels[1] == 1:
#             lbl = 0
#         else:
#             lbl = 1
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, {"labels": lbl, "all_true": labels.sum() == len(self.classes)}
#
#     def __len__(self):
#         return self.data_len


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, transform=None):
        """Set the path for images.

        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.transform = transform
        self.ids = []
        ids = list(self.coco.imgs.keys())

        # # # TOP:
        # ((1, 2, 3), ['person', 'bicycle', 'car'], [2, 2, 2], 45, [77, 314, 7])
        # ((1, 3, 4), ['person', 'car', 'motorcycle'], [2, 2, 2], 51, [308, 77, 8])
        # ((1, 3, 5), ['person', 'car', 'airplane'], [2, 2, 2], 3, [356, 28, 6])
        # ((1, 3, 6), ['person', 'car', 'bus'], [6, 6, 6], 82, [277, 61, 20])
        # ((1, 3, 8), ['person', 'car', 'truck'], [17, 17, 17], 101, [258, 58, 51])
        # ((1, 44, 47), ['person', 'bottle', 'cup'], [18, 18, 18], 74, [133, 143, 54])
        # ((1, 47, 51), ['person', 'cup', 'bowl'], [19, 19, 19], 60, [157, 72, 58])
        # ((1, 47, 67), ['person', 'cup', 'dining table'], [24, 24, 24], 120, [97, 130, 74])
        # ((1, 62, 67), ['person', 'chair', 'dining table'], [29, 29, 29], 152, [215, 98, 89])
        # # # # #
        # ([[1, 3], [6, 8]], [['person', 'car'], ['bus', 'truck']], 27)
        # self.classes = [(1, 3), (6, 8)]
        # ([[1, 47], [62, 67]], [['person', 'cup'], ['chair', 'dining table']], 74)
        # self.classes = [(1, 47), (62, 67)]
        # ((4, 1, 3), ['motorcycle', 'person', 'car'], 217, 1160, [1626, 217, 7359])
        self.classes = [1, 3]
        # self.classes = [(1, 62), (62, 67)]
        self.classes = sorted(self.classes)
        self.class_names = [self.coco.loadCats(_c)[0]["name"] for _c in self.classes]
        print([self.coco.loadCats(_c)[0]["name"] for _c in self.classes])

        all_positive_cats = []
        partial_positive_cats = [[] for _ in self.classes]
        for id in ids:
            ann_ids = self.coco.getAnnIds(imgIds=id)
            coco_annotations = self.coco.loadAnns(ann_ids)
            categories = [ann["category_id"] for ann in coco_annotations]
            if set(self.classes) <= set(categories):
                all_positive_cats.append(id)
            elif len(set(self.classes).intersection(categories)) == 1:
                for ind, c in enumerate(self.classes):
                    if c in categories:
                        partial_positive_cats[ind].append(id)
        dataset_frac_len = min([len(all_positive_cats) // 2, len(partial_positive_cats[0]), len(partial_positive_cats[1])])
        shuffle(all_positive_cats)
        for p in partial_positive_cats:
            shuffle(p)
        all_positive_cats = all_positive_cats[:dataset_frac_len * 2]
        for p in range(len(partial_positive_cats)):
            partial_positive_cats[p] = partial_positive_cats[p][:dataset_frac_len]
        print(len(all_positive_cats), len(partial_positive_cats[0]), len(partial_positive_cats[1]))

        self.ids = all_positive_cats + partial_positive_cats[0] + partial_positive_cats[1]
        shuffle(self.ids)
        self.data_len = len(self.ids)

    def __getitem__(self, index):
        """Returns one data pair (image and class labels)."""
        ann_ids = self.coco.getAnnIds(imgIds=self.ids[index])
        coco_annotations = self.coco.loadAnns(ann_ids)
        cats = [ann["category_id"] for ann in coco_annotations]
        path = self.coco.loadImgs(self.ids[index])[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # plt.imshow(img)
        # self.coco.showAnns(coco_annotations)
        # fig = plt.gcf()
        # img = fig2img(fig)

        if set(self.classes) <= set(cats):
            lbl = 1
        else:
            lbl = 0

        if self.transform is not None:
            img = self.transform(img)

        return img, {"labels": lbl, "all_true": set(self.classes) <= set(cats)}

    def __len__(self):
        return self.data_len


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# class CocoDataset(data.Dataset):
#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
#
#     def __init__(self, root, json, transform=None):
#         """Set the path for images.
#
#         Args:
#             root: image directory.
#             json: coco annotation file path.
#             transform: image transformer.
#         """
#         self.root = root
#         self.coco = COCO(json)
#         self.transform = transform
#         self.ids = []
#         ids = list(self.coco.imgs.keys())
#
#         # coco = self.coco.COCO(json)
#         # self.class_names = classes = [["motorcycle", "person"], ["person", "car"]]
#         # catIds = coco.getCatIds(catNms=[_c for c in classes for _c in c])
#         # imgIds = coco.getImgIds(catIds=catIds)
#         # imgDict = coco.loadImgs(imgIds)
#
#         # # # TOP:
#         # ((1, 2, 3), ['person', 'bicycle', 'car'], [2, 2, 2], 45, [77, 314, 7])
#         # ((1, 3, 4), ['person', 'car', 'motorcycle'], [2, 2, 2], 51, [308, 77, 8])
#         # ((1, 3, 5), ['person', 'car', 'airplane'], [2, 2, 2], 3, [356, 28, 6])
#         # ((1, 3, 6), ['person', 'car', 'bus'], [6, 6, 6], 82, [277, 61, 20])
#         # ((1, 3, 8), ['person', 'car', 'truck'], [17, 17, 17], 101, [258, 58, 51])
#         # ((1, 44, 47), ['person', 'bottle', 'cup'], [18, 18, 18], 74, [133, 143, 54])
#         # ((1, 47, 51), ['person', 'cup', 'bowl'], [19, 19, 19], 60, [157, 72, 58])
#         # ((1, 47, 67), ['person', 'cup', 'dining table'], [24, 24, 24], 120, [97, 130, 74])
#         # ((1, 62, 67), ['person', 'chair', 'dining table'], [29, 29, 29], 152, [215, 98, 89])
#         # # # # #
#         # ([[1, 3], [6, 8]], [['person', 'car'], ['bus', 'truck']], 27)
#         # self.classes = [(1, 3), (6, 8)]
#         # ([[1, 47], [62, 67]], [['person', 'cup'], ['chair', 'dining table']], 74)
#         # self.classes = [(1, 47), (62, 67)]
#         # ((4, 1, 3), ['motorcycle', 'person', 'car'], 217, 1160, [1626, 217, 7359])
#         self.classes = [(4, 1), (1, 3)]
#         # self.classes = [(1, 62), (62, 67)]
#         self.classes = [sorted(c) for c in self.classes]
#         self.class_names = [[self.coco.loadCats(_c)[0]["name"] for _c in c] for c in self.classes]
#         print([[self.coco.loadCats(_c)[0]["name"] for _c in c] for c in self.classes])
#         all_classes = set([c for _class in self.classes for c in _class])
#         non_classes = [comb for comb in list(combinations(all_classes, 2)) if sorted(comb) not in self.classes]
#
#         all_positive_cats = []
#         partial_positive_cats = [[] for _ in self.classes]
#         partial_negative_cats = [[] for _ in non_classes]
#         for i, id in enumerate(ids):
#             ann_ids = self.coco.getAnnIds(imgIds=id)
#             coco_annotations = self.coco.loadAnns(ann_ids)
#             categories = [ann["category_id"] for ann in coco_annotations]
#             if all_classes <= set(categories):
#                 all_positive_cats.append(id)
#             elif len(all_classes.intersection(categories)) == 2:
#                 for ind, c in enumerate(self.classes):
#                     if set(c) <= set(categories):
#                         partial_positive_cats[ind].append(id)
#                 for ind, c in enumerate(non_classes):
#                     print(set(c))
#                     if set(c) <= set(categories):
#                         partial_negative_cats[ind].append(id)
#         frac = (len(partial_positive_cats) + 1) / len(partial_negative_cats)
#         dataset_frac_len = int(min([min([len(p) for p in partial_positive_cats]),
#                                min([len(p) for p in partial_negative_cats]) // frac]))
#         shuffle(all_positive_cats)
#         for p in partial_positive_cats:
#             shuffle(p)
#         for p in partial_negative_cats:
#             shuffle(p)
#         all_positive_cats = all_positive_cats[:min([dataset_frac_len, len(all_positive_cats)])]
#         for p in range(len(partial_positive_cats)):
#             partial_positive_cats[p] = partial_positive_cats[p][:dataset_frac_len]
#         for p in range(len(partial_negative_cats)):
#             partial_negative_cats[p] = partial_negative_cats[p][:int(dataset_frac_len * frac)]
#         print(len(all_positive_cats), len(partial_positive_cats[0]), len(partial_positive_cats[1]),
#               len(partial_negative_cats[0]))
#
#         self.ids = all_positive_cats + partial_positive_cats[0] + partial_positive_cats[1] + partial_negative_cats[0]
#         shuffle(self.ids)
#         self.data_len = len(self.ids)
#
#
#     def __getitem__(self, index):
#         """Returns one data pair (image and class labels)."""
#         ann_ids = self.coco.getAnnIds(imgIds=self.ids[index],
#                                       catIds=[c for classes in self.classes for c in classes],
#                                       iscrowd=None)
#         coco_annotations = self.coco.loadAnns(ann_ids)
#         cats = [ann["category_id"] for ann in coco_annotations]
#         path = self.coco.loadImgs(self.ids[index])[0]['file_name']
#         img = Image.open(os.path.join(self.root, path)).convert("RGB")
#         # plt.imshow(img)
#         # self.coco.showAnns(coco_annotations)
#         # fig = plt.gcf()
#         # img = fig2img(fig)
#
#         # mask = self.coco.annToMask(coco_annotations[0])
#         # for i in range(len(coco_annotations)):
#         #     mask = mask | self.coco.annToMask(coco_annotations[i])
#         # imshow(mask)
#
#         labels = torch.zeros(len(self.classes))
#         for i, c in enumerate(self.classes):
#             if set(c) <= set(cats):
#                 labels[i] = 1
#         if labels.sum() == len(self.classes):
#             lbl = 0
#         elif labels[0] == 1:
#             lbl = 0
#         elif labels[1] == 1:
#             lbl = 0
#         else:
#             lbl = 1
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, {"labels": lbl, "all_true": labels.sum() == len(self.classes)}
#
#     def __len__(self):
#         return self.data_len