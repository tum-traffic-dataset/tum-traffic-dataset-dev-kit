import os
import glob
import json
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou, ConfusionMatrix
from ultralytics.utils import LOGGER, colorstr, ops
from ultralytics.utils.plotting import plot_images
import cv2
from pathlib import Path

CATEGORY = ["car", "truck", "trailer", "van", "motorcycle", "bus", "pedestrian", "bicycle", "emergency vehicle",
            "other"]


def parse_input_files(input_folder_or_file_path):
    if os.path.isdir(input_folder_or_file_path):
        labels_file_list = sorted(glob.glob(os.path.join(input_folder_or_file_path, "*.json")))
    else:
        labels_file_list = [input_folder_or_file_path]

    print("--- amount of files: ", len(labels_file_list))

    file_name_to_labels = {}

    for label_file in labels_file_list:
        with open(label_file, "r") as input_file:
            json_data = json.load(input_file)
        if "openlabel" in json_data:
            file_name = label_file.split("/")[-1]
            category_list = []
            bbox_list = []
            poly2d_list = []
            prediction_conf_list = []

            for frame_id, frame_obj in json_data["openlabel"]["frames"].items():
                for object_id, label in frame_obj["objects"].items():
                    category = CATEGORY.index(label["object_data"]["type"].lower())  # checked: gt, yolo_predict

                    # extract poly2d = [x1, y1, x2, y2, ...]                   # checked: gt, yolo_predict (both has 1 with "name": "mask")
                    poly2d = label["object_data"]["poly2d"][0]["val"]

                    # extract bbox = x_center, y_center, width, height         # checked: gt, yolo_predict (both has 1 with "name": "full_bbox")
                    if "bbox" in label["object_data"]:
                        bbox = label["object_data"]["bbox"][0]["val"]

                        # extract prediction confidence
                        if "attributes" in label["object_data"]["bbox"][0]:
                            if "num" in label["object_data"]["bbox"][0]["attributes"]:
                                prediction_conf_list.append(
                                    label["object_data"]["bbox"][0]["attributes"]["num"][0]["confidence"])
                    else:
                        # calculate bbox from poly2d
                        x_min = min(poly2d[::2])
                        x_max = max(poly2d[::2])
                        y_min = min(poly2d[1::2])
                        y_max = max(poly2d[1::2])
                        width = x_max - x_min
                        height = y_max - y_min
                        x_center = x_min + width / 2.0
                        y_center = y_min + height / 2.0
                        bbox = [int(x_center), int(y_center), int(width), int(height)]

                    category_list.append(category)
                    bbox_list.append(bbox)
                    poly2d_list.append(poly2d)

            if len(prediction_conf_list) > 0:
                file_name_to_labels[file_name] = {
                    "categories": np.array(category_list),
                    "bboxes": np.array(bbox_list),
                    "poly2ds": np.array(poly2d_list, dtype=object),
                    "confidences": np.array(prediction_conf_list),
                }
            else:
                file_name_to_labels[file_name] = {
                    "categories": np.array(category_list),
                    "bboxes": np.array(bbox_list),
                    "poly2ds": np.array(poly2d_list, dtype=object),
                }

    return file_name_to_labels


def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (torch.Tensor): Predicted class indices of shape(N,).
        true_classes (torch.Tensor): Target class indices of shape(M,).
        iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
        use_scipy (bool): Whether to use scipy for matching (more precise).

    Returns:
        (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
    """
    # Dx10 matrix, where D - detections, 10 - IoU thresholds
    correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
    # LxD matrix where L - labels (rows), D - detections (columns)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(self.iouv.cpu().tolist()):
        if use_scipy:
            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
            import scipy  # scope import to avoid importing for all commands
            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
        else:
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)


def polygon2d_to_mask(width, height, poly2d):
    mask = np.zeros((height, width), dtype=np.uint8)
    num_points = len(poly2d) // 2
    if num_points:
        polygon_array = np.array(poly2d).reshape(num_points, 2)
        cv2.fillPoly(mask, [polygon_array.astype(np.int32)], color=1)
    else:
        print("---- empy mask")
        return []
    return mask


class Yolov8_SegmentationValidator():

    def __init__(self, gt_data, pred_data, args):
        self.task = 'segment'
        self.device = 'cpu'
        self.save_dir = Path(args.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = SegmentMetrics(save_dir=self.save_dir)

        LOGGER.info(('%22s' + '%11s' * 10) % (
            'Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Mask(P', 'R', 'mAP50', 'mAP50-95)'))

        ## init_metrics
        self.names = {
            0: 'car',
            1: 'truck',
            2: 'trailer',
            3: 'van',
            4: 'motorcycle',
            5: 'bus',
            6: 'person',
            7: 'bicycle',
            8: 'emergency vehicle',
            9: 'other'
        }
        self.nc = len(self.names)
        self.metrics.names = self.names
        self.metrics.plot = args.plots
        self.conf = 0.001
        # self.conf = 0.8

        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf)
        self.seen = 0
        self.stats = []

        self.gt_data = gt_data
        self.pred_data = pred_data
        self.ori_shape = [1200, 1920]
        self.height = 1216
        self.width = 1920
        self.plots = args.plots
        self.image_folder_path = args.image_folder_path
        self.amount_image_print = 3

    def validate(self):
        image_counter = 0
        for img_name, groundTruth in self.gt_data.items():

            ## groundTruth
            groundTruth["categories"] = torch.tensor(groundTruth["categories"], device=self.device, dtype=torch.float32)
            groundTruth["bboxes"] = torch.tensor(groundTruth["bboxes"], device=self.device)
            groundTruth["masks"] = []
            for poly2d in groundTruth["poly2ds"]:
                groundTruth["masks"].append(polygon2d_to_mask(self.width, self.height, poly2d))
            groundTruth["masks"] = torch.tensor(np.array(groundTruth["masks"]), device=self.device, dtype=torch.float32)

            ## preprocess, inference, postprocess -> prediction
            prediction = self.pred_data[img_name]
            prediction["categories"] = torch.tensor(prediction["categories"], device=self.device, dtype=torch.float32)
            prediction["bboxes"] = torch.tensor(prediction["bboxes"], device=self.device)
            prediction["masks"] = []
            for poly2d in prediction["poly2ds"]:
                prediction["masks"].append(polygon2d_to_mask(self.width, self.height, poly2d))

            prediction["masks"] = torch.tensor(np.array(prediction["masks"]), device=self.device, dtype=torch.float32)
            if "confidences" in prediction:
                prediction["confidences"] = torch.tensor(prediction["confidences"], device=self.device,
                                                         dtype=torch.float32)
            else:
                prediction["confidences"] = torch.ones(prediction["bboxes"].shape[0], dtype=torch.float32)

            nl = groundTruth["categories"].shape[0]  # number of labels
            npr = prediction["categories"].shape[0]  # number of predictions
            correct_masks = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    print("---- npr==0 and nl!=0")
                    self.stats.append((correct_bboxes, correct_masks, *torch.zeros((2, 0), device=self.device),
                                       groundTruth["categories"]))
                    if self.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=groundTruth["categories"])
                continue

            # Evaluate
            if nl:
                # process bbox
                target_box = ops.xywh2xyxy(groundTruth[
                                               "bboxes"])  # * torch.tensor((self.width, self.height, self.width, self.height), device=self.device)
                pred_bbox = ops.xywh2xyxy(prediction[
                                              "bboxes"])  # * torch.tensor((self.width, self.height, self.width, self.height), device=self.device) #tensor shape (M, 4) representing M bounding boxes.
                labelsn = torch.cat((groundTruth["categories"].unsqueeze(1), target_box),
                                    dim=1)  # (array[N, 5]), class, x1, y1, x2, y2
                predn = torch.cat(
                    (pred_bbox, prediction["confidences"].unsqueeze(1), prediction["categories"].unsqueeze(1)),
                    dim=1)  # (Array[N, 6]), (x1, y1, x2, y2, conf, class).

                # iou = box_iou(target_box, pred_bbox)
                iou = box_iou(labelsn[:, 1:], predn[:, :4])
                # print("--- box_iou", iou)
                # correct_bboxes = match_predictions(self, prediction["categories"], groundTruth["categories"], iou)
                correct_bboxes = match_predictions(self, predn[:, 5], labelsn[:, 0], iou)

                ## process mask
                if groundTruth["masks"].shape[1:] != prediction["masks"].shape[1:]:
                    print("---- interpolate masks")
                    groundTruth["masks"] = \
                        F.interpolate(groundTruth["masks"][None], prediction["masks"].shape[1:], mode='bilinear',
                                      align_corners=False)[0]
                    groundTruth["masks"] = groundTruth["masks"].gt_(0.5)

                # print("--- mask_iou", iou)
                iou = mask_iou(groundTruth["masks"].view(groundTruth["masks"].shape[0], -1),
                               prediction["masks"].view(prediction["masks"].shape[0], -1))
                # correct_masks = match_predictions(self, prediction["categories"], groundTruth["categories"], iou)
                correct_masks = match_predictions(self, predn[:, 5], labelsn[:, 0], iou)

                ## process confusion_matrix
                if self.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)

            # Append correct_masks, correct_bboxes, pconf, pcls, tcls
            self.stats.append((correct_bboxes, correct_masks, prediction["confidences"], prediction["categories"],
                               groundTruth["categories"]))

            if args.plots and image_counter < self.amount_image_print:
                # plot_val_samples
                torch_img = np.array(cv2.imread(self.image_folder_path + img_name.rsplit('.', 1)[0] + '.jpg'))
                torch_img = torch.from_numpy(torch_img.transpose((2, 0, 1))).unsqueeze(0)  # Add a batch dimension

                plot_images(
                    torch_img,
                    torch.zeros(nl, device='cuda:0'),  # batch_idx
                    groundTruth["categories"].unsqueeze(1).squeeze(-1),
                    groundTruth["bboxes"],
                    groundTruth['masks'],
                    fname=self.save_dir / f'val_{img_name}_labels.jpg',
                    names=self.names
                )

                # plot_predictions
                plot_images(
                    torch_img,
                    torch.zeros(npr, device='cuda:0'),  # batch_idx
                    prediction["categories"].unsqueeze(1).squeeze(-1),
                    torch.cat((prediction["bboxes"], prediction["confidences"].unsqueeze(1)), dim=1),
                    prediction['masks'],
                    fname=self.save_dir / f'val_{img_name}_pred.jpg',
                    names=self.names,
                    # on_plot = self.on_plot
                )

            image_counter += 1

            # print("---labelsn", labelsn.shape, labelsn)
            # print("---predn", predn.shape, predn)

        ##get_stats
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        # print("---self.conf", self.conf)
        if len(stats) and stats[0].any():
            self.metrics.process(*stats)

        self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        stats = self.metrics.results_dict

        ## finalize_metrics
        self.metrics.confusion_matrix = self.confusion_matrix

        ## print_results
        # all
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f'WARNING ⚠️ no labels found in {self.task} set, can not compute metrics without labels')

        # Print results per class
        if self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,  # on_plot=self.on_plot
                                           )

        ## log saved folder path
        if self.plots:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def calculate_mIoU(self):
        image_counter = 0
        iou_list = [[] for _ in range(len(self.names))]

        for img_name, groundTruth in self.gt_data.items():
            prediction = self.pred_data[img_name]
            # prepare
            class_wise_gt_mask = [np.zeros((self.width, self.height), dtype=np.uint8) for _ in range(len(self.names))]
            class_wise_predicted_mask = [np.zeros((self.width, self.height), dtype=np.uint8) for _ in
                                         range(len(self.names))]

            for poly2d, category in zip(groundTruth["poly2ds"], groundTruth["categories"]):
                mask = polygon2d_to_mask(self.width, self.height, poly2d)
                # add this mask to
                class_wise_gt_mask[int(category)] &= mask

            for poly2d, category in zip(prediction["poly2ds"], prediction["categories"]):
                mask = polygon2d_to_mask(self.width, self.height, poly2d)
                # add this mask to
                class_wise_predicted_mask[int(category)] &= mask

            image_counter += 1

            for class_idx in range(10):
                SMOOTH = 1e-10  # based on c2f-seg/utils/evaluation.py
                intersection = np.logical_and(class_wise_gt_mask[class_idx], class_wise_predicted_mask[class_idx])
                union = np.logical_or(class_wise_gt_mask[class_idx], class_wise_predicted_mask[class_idx])
                iou = (intersection.sum() + SMOOTH) / (union.sum() + SMOOTH)
                iou_list[class_idx].append(iou)
                # print(f"- class {classes[class_idx]} visible iou = {iou}")

        print()
        print("Done with", image_counter, "frames. Visible meanIoU of")
        mean_iou_list = []
        for class_idx in range(len(self.names)):
            mean_iou = np.mean(iou_list[class_idx])
            mean_iou_list.append(mean_iou)
            print(f"{self.names[class_idx]} = {mean_iou}")
        print("---- Visible mean IoU = ", np.mean(mean_iou_list))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--image_folder_path", type=str, help="Path to folder containing images",
                            default="/path/to/TUMTraf/test/images/")
    arg_parser.add_argument("--path_to_ground_truth", type=str, help="Path to folder or file containing gt labels",
                            default="/path/to/TUMTraf/test/labels_openlabel")
    arg_parser.add_argument("--path_to_predictions", type=str,
                            help="Path to folder or file containing predicted labels",
                            default="/path/to/inference_result/val_train/openlabel")
    arg_parser.add_argument("--prediction_format", type=str, help="possible values: openlabel", default="openlabel")
    arg_parser.add_argument("--plots", type=bool, help="Bool: Plot the result or not", default=True)
    arg_parser.add_argument("--save_dir", type=str, help="",
                            default="/path/to/dataset-dev-kit/src/eval/evaluation_2d_result/")
    args = arg_parser.parse_args()

    gt_data = parse_input_files(args.path_to_ground_truth)

    if args.prediction_format == "openlabel":
        pred_data = parse_input_files(args.path_to_predictions)
    else:
        raise ValueError("Unknown prediction format: {}".format(args.prediction_format))

    # YOLOv8 mAP
    segmentationValidator = Yolov8_SegmentationValidator(gt_data, pred_data, args)
    # segmentationValidator.validate()

    # mIoU
    segmentationValidator.calculate_mIoU()
