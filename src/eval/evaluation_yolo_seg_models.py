import argparse
from ultralytics import YOLO
import os
import subprocess
import numpy as np
import cv2
import yaml


def evaluate_mIoU(model, args, visualize):
    classes = ["car", "truck", "trailer", "van", "motorcycle", "bus", "person", "bicycle", "emergency vehicle", "other"]

    with open(args.data_yaml, 'r') as file:
        yaml_content = yaml.safe_load(file)

    # If 'val' is a path, you can join it with the base path
    if yaml_content.get('val', '') and yaml_content.get('path', ''):
        test_txt_path = os.path.join(yaml_content.get('path', ''), yaml_content.get('val', ''))
        frame_list = []
        with open(test_txt_path, 'r') as f:
            for line in f:
                frame_list.append(line.strip())

        results = model(frame_list, stream=True, conf=args.conf, imgsz=args.imgsz, batch=1)

        frame_counter = 0
        iou_list = [[] for _ in range(len(classes))]
        try:
            for r in results:
                class_wise_gt_mask = [np.zeros((r.orig_shape[0], r.orig_shape[1]), dtype=np.uint8) for _ in
                                      range(len(classes))]
                class_wise_predicted_mask = [np.zeros((r.orig_shape[0], r.orig_shape[1]), dtype=np.uint8) for _ in
                                             range(len(classes))]

                label_dir = os.path.dirname(r.path.replace("images", "labels"))
                label_filename = os.path.splitext(os.path.basename(r.path))[0] + ".txt"
                label_path = os.path.join(label_dir, label_filename)

                with open(label_path, "r") as f:
                    for line in f:
                        polygon = []
                        for p in line.split()[1:]:
                            polygon.append(float(p))
                        polygon = [int(p * r.orig_shape[1]) if i % 2 == 0 else int(p * r.orig_shape[0]) for i, p in
                                   enumerate(polygon)]
                        polygon = np.array(polygon).reshape((-1, 2)).reshape((-1, 1, 2)).astype(np.int32)
                        cv2.fillPoly(class_wise_gt_mask[int(line.split()[0])], [polygon], color=1)

                if r.masks:
                    for bbox_result, mask_result in zip(r.boxes, r.masks):
                        if mask_result.xyn[0].size != 0:
                            polygon_points = mask_result.xyn[0] * np.array([r.orig_shape[1], r.orig_shape[0]])
                            polygon_points = polygon_points.reshape((-1, 1, 2)).astype(np.int32)
                            cv2.fillPoly(class_wise_predicted_mask[int(bbox_result.cls.to('cpu').item())],
                                         [polygon_points], color=1)

                frame_counter += 1

                # calculate_iou(gt_mask, predicted_mask):
                # print("Frame", frame_counter, "has:")
                for class_idx in range(10):
                    SMOOTH = 1e-10  # based on c2f-seg/utils/evaluation.py
                    intersection = np.logical_and(class_wise_gt_mask[class_idx], class_wise_predicted_mask[class_idx])
                    union = np.logical_or(class_wise_gt_mask[class_idx], class_wise_predicted_mask[class_idx])
                    iou = (intersection.sum() + SMOOTH) / (union.sum() + SMOOTH)
                    iou_list[class_idx].append(iou)
                    # print(f"- class {classes[class_idx]} visible iou = {iou}")

                # if visualize:
                #    original_image = cv2.imread(r.path)
                #    if original_image is None:
                #        raise ValueError("Could not read the image")
                #    masked_image = original_image.copy()
                #    masked_image[gt_mask == 1] = [0, 255, 0]
                #    masked_image[predicted_mask == 1] = [0, 0, 255]
                #    cv2.imwrite('masked_image.jpg', masked_image)

        except Exception as ex:
            print("An error occurred:", ex)

        print()
        print("Done with", frame_counter, "frames. Visible meanIoU of")
        mean_iou_list = []
        for class_idx in range(len(classes)):
            mean_iou = np.mean(iou_list[class_idx])
            mean_iou_list.append(mean_iou)
            print(f"{classes[class_idx]} = {mean_iou}")
        print("---- Visible mean IoU = ", np.mean(mean_iou_list))

    else:
        print("No 'val' or 'path' key found in the YAML file.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--yolo_version", type=str, help="possible values: yolov8, yolov7", default="yolov7")
    arg_parser.add_argument("--conf", type=float, help="", default=0.25)  # 0.25, 0.001
    arg_parser.add_argument("--imgsz", type=int, help="", default=640)  # 1920, 1280, 640
    arg_parser.add_argument("--path_to_model_weight", type=str, help="Path to model .pt weight",
                            default="/path/to/yolov8/runs/segment/train4/weights/best.pt")
    arg_parser.add_argument("--data_yaml", type=str, help="Path to data.yaml file",
                            default="/path/to/TUMTraf/data.yaml")
    arg_parser.add_argument("--show_meanIoU", type=str, help="Evaluate mIoU or not", default=True)
    args = arg_parser.parse_args()

    # if evaluate with model trained on COCO, set label folder labels = test/labels_coco_cls_to_yolo_cls
    if args.yolo_version == "yolov8":
        model = YOLO(args.path_to_model_weight)
        metrics = model.val(data=args.data_yaml, imgsz=args.imgsz, conf=args.conf, batch=1)

        if args.show_meanIoU:
            evaluate_mIoU(model, args, visualize=False)

    elif args.yolo_version == "yolov7":
        command = f"python /path/to/yolov7_seg/segment/val.py \
                    --weights yolov7x-seg.pt \
                    --data /path/to/TUMTraf/coco_data.yaml \
                    --batch-size=16 \
                    --conf={args.conf} --imgsz={args.imgsz} --verbose"
        subprocess.run(command, shell=True)

    else:
        raise ValueError("Unknown yolo version: {}".format(args.yolo_version))

