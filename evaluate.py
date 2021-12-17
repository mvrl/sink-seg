import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from model import Unet
from data_factory import get_data
from config import cfg


def get_confusion_matrix_binary(label, pred, size, num_class, ignore=-1):
    """
    Compute binary confusion matrix.
    Code adopted from: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/pytorch-v1.1/lib/utils/utils.py
    """
    seg_pred = np.asarray(pred.cpu().numpy(), dtype=np.uint8)

    seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]],
                        dtype=np.int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    return confusion_matrix


def evaluate():
    """
    Loads a saved checkpoint from disk, finds optimum threshold using the val set, and computes test set results.
    All results are saved to the disk.
    """

    # model type: 'best' (based on lowest val loss) or 'end'
    eval_mode = 'best'

    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        raise ValueError(
            'The directory with trained model does not exist! Make sure cfg.train.out_dir in config.py has the correct directory name'
        )

    from model import Unet
    model = Unet(in_channels=cfg.data.input_channels,
                 out_channels=2,
                 feature_reduction=4,
                 norm_type=cfg.model.norm_type)
    model.to('cuda:0')

    # which checkpoint to load: best (lowest val loss) or the one saved at the end of training
    if eval_mode == 'best':
        fname = os.path.join(out_dir, 'model_dict.pth')
    else:
        fname = os.path.join(out_dir, 'model_dict_end.pth')

    model.load_state_dict(torch.load(fname))
    model.eval()

    # get dataloaders
    cfg.train.batch_size = 1
    # enable padding in the evaluation
    cfg.data.eval_pad = True
    cfg.train.shuffle = False
    _, data_loader_val, data_loader_test = get_data(cfg)

    num_classes = 2
    """ Find out best threshold on the val set """
    print('\nFinding out the best threshold on val loader')
    best_iou = torch.zeros(1).cuda()
    best_acc = torch.zeros(1).cuda()
    best_threshold = torch.zeros(1).cuda()

    acc_log = []
    iou_log = []
    thresh_log = []

    iou_new = []
    acc_new = []

    thresholds = np.linspace(start=0, stop=1, num=21)
    for j in range(thresholds.shape[0]):
        threshold = thresholds[j]

        confusion_matrix = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for i, data in enumerate(data_loader_val):
                shaded = data[0].cuda()
                dem = data[1].cuda().unsqueeze(1)
                naip_image = data[2].cuda()
                labels = data[3].long().cuda()
                dem_dxy = data[5].cuda()
                dem_dxy_pre = data[6].cuda().unsqueeze(1)

                predictions = model(shaded, dem, naip_image, dem_dxy,
                                    dem_dxy_pre)

                predictions = torch.softmax(predictions, dim=1)

                if predictions.shape[2] == 420:
                    starty = 0
                    endy = 400
                else:
                    starty = 40
                    endy = 440

                if predictions.shape[3] == 420:
                    startx = 0
                    endx = 400
                else:
                    startx = 40
                    endx = 440
                predictions = predictions[:, :, starty:endy, startx:endx]
                dem = dem[:, :, starty:endy, startx:endx]
                labels = labels[:, starty:endy, startx:endx]

                pred_final = (predictions[:, 1, :, :] > threshold).long()

                confusion_matrix += get_confusion_matrix_binary(
                    label=labels,
                    pred=pred_final,
                    size=cfg.data.cutout_size,
                    num_class=num_classes)

        # compute metrics from the confusion matrix
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum() / pos.sum()
        mean_acc = (tp / np.maximum(1.0, pos)).mean()
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        sinkhole_iou = IoU_array[1]

        iou_new.append(IoU_array[1] * 100.0)
        acc_new.append(mean_acc)
        thresh_log.append(threshold)

        print('iou: ', sinkhole_iou, ' threshold: ', threshold)

        if sinkhole_iou > best_iou:
            best_iou = sinkhole_iou
            best_threshold = threshold

    print('==== finished analysis ==== best iou: ', best_iou.item(),
          ' best threshold: ', best_threshold)

    # Saving metrics vs. thresholds plot to disk
    plt.figure()
    plt.plot(thresh_log, iou_new)
    plt.ylabel('Sinkhole IoU (%)')
    plt.xlabel('threshold')
    name_string = 'threshold_mertics_val.png'
    fname = os.path.join(out_dir, name_string)
    plt.savefig(fname)
    plt.close()

    # Saving metrics vs. thresholds details disk
    name_string = 'threshold_vs_metrics_val.txt'
    fname = os.path.join(out_dir, name_string)
    with open(os.path.join(fname), 'w') as result_file:
        result_file.write('Results of metrics vs threshold... \n')
        result_file.write('thresh_log: ')
        result_file.write(str(thresh_log))
        result_file.write('\nbest threshold: ')
        result_file.write(str(best_threshold))
        result_file.write('\nIoU: ')
        result_file.write(str(iou_new))
        result_file.write('\nAccuracy: ')
        result_file.write(str(acc_new))

    # save best threshold
    name_string = 'best_threshold.txt'
    fname = os.path.join(out_dir, name_string)
    with open(os.path.join(fname), 'w') as result_file:
        result_file.write(str(best_threshold))
    """ Evaluate on the test set """
    print('\nComputing test set metrics')
    iou_threshold = torch.zeros(1).cuda()
    acc_threshold = torch.zeros(1).cuda()

    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for i, data in enumerate(data_loader_test):
            shaded = data[0].cuda()
            dem = data[1].cuda().unsqueeze(1)
            naip_image = data[2].cuda()
            labels = data[3].long().cuda()
            dem_dxy = data[5].cuda()
            dem_dxy_pre = data[6].cuda().unsqueeze(1)

            predictions = model(shaded, dem, naip_image, dem_dxy, dem_dxy_pre)

            if predictions.shape[2] == 420:
                starty = 0
                endy = 400
            else:
                starty = 40
                endy = 440

            if predictions.shape[3] == 420:
                startx = 0
                endx = 400
            else:
                startx = 40
                endx = 440
            predictions = predictions[:, :, starty:endy, startx:endx]
            dem = dem[:, :, starty:endy, startx:endx]
            labels = labels[:, starty:endy, startx:endx]

            pred_final = torch.softmax(predictions, dim=1)
            pred_final = (pred_final[:, 1, :, :] > best_threshold).long()

            confusion_matrix += get_confusion_matrix_binary(
                label=labels,
                pred=pred_final,
                size=cfg.data.cutout_size,
                num_class=num_classes)

    # compute metrics from the confusion matrix
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    sinkhole_iou = IoU_array[1]

    # Print details
    print('Results on the test set:')
    print('pixel acc: ', pixel_acc)
    print('mean acc: ', mean_acc)
    print('IoU full: ', IoU_array)
    print('Sinkhole IoU: ', IoU_array[1])
    print('m IoU: ', mean_IoU)
    """ Make full size predictions """
    print('\nMaking full-size predictions on the test set')

    data_mode = 'test'
    if data_mode == 'val':
        data_loader = data_loader_val
        size_y = 9200
        size_x = 2800
    elif data_mode == 'test':
        data_loader = data_loader_test
        size_y = 9600
        size_x = 14000

    true_label = torch.zeros(size_y, size_x)
    inp_dem = torch.zeros(size_y, size_x)
    pred = torch.zeros(size_y, size_x)
    pred_raw = torch.zeros(size_y, size_x)

    # initialize full-size tiles for various thresholds: 0.3, 0.6, and 0.9
    pred_30 = torch.zeros(size_y, size_x)
    pred_60 = torch.zeros(size_y, size_x)
    pred_90 = torch.zeros(size_y, size_x)

    cutout_size = cfg.data.cutout_size

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            shaded = data[0].cuda()
            dem = data[1].cuda().unsqueeze(1)
            naip_image = data[2].cuda()
            labels = data[3].long().cuda()
            idx = data[4].detach().cpu().numpy()
            dem_dxy = data[5].cuda()
            dem_dxy_pre = data[6].cuda().unsqueeze(1)

            num_columns = 7 if data_mode == 'val' else 35
            row, col = divmod(idx, num_columns)

            left = col * cutout_size[0]
            upper = row * cutout_size[1]
            right = (col + 1) * cutout_size[0]
            lower = (row + 1) * cutout_size[1]

            predictions = model(shaded, dem, naip_image, dem_dxy, dem_dxy_pre)

            predictions = torch.softmax(predictions, dim=1)

            if predictions.shape[2] == 420:
                starty = 0
                endy = 400
            else:
                starty = 40
                endy = 440

            if predictions.shape[3] == 420:
                startx = 0
                endx = 400
            else:
                startx = 40
                endx = 440
            predictions = predictions[:, :, starty:endy, startx:endx]
            dem = dem[:, :, starty:endy, startx:endx]
            labels = labels[:, starty:endy, startx:endx]

            true_label[upper[0]:lower[0], left[0]:right[0]] = labels

            pred_max = torch.argmax(predictions, dim=1)

            pred[upper[0]:lower[0],
                 left[0]:right[0]] = pred_max  # prediction tile

            pred_raw[upper[0]:lower[0], left[0]:right[0]] = predictions[:,
                                                                        1, :, :]

            # input
            inp_dem[upper[0]:lower[0], left[0]:right[0]] = dem
            pred_30[upper[0]:lower[0],
                    left[0]:right[0]] = 1.0 * (predictions[:, 1, :, :] > 0.30)
            pred_60[upper[0]:lower[0],
                    left[0]:right[0]] = 1.0 * (predictions[:, 1, :, :] > 0.60)
            pred_90[upper[0]:lower[0],
                    left[0]:right[0]] = 1.0 * (predictions[:, 1, :, :] > 0.90)

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 5, 1)
    plt.imshow(pred_30)
    plt.title('t=0.3')
    plt.subplot(1, 5, 2)
    plt.imshow(pred_60)
    plt.title('t=0.6')
    plt.subplot(1, 5, 3)
    plt.imshow(pred_90)
    plt.title('t=0.9')
    plt.subplot(1, 5, 4)
    plt.imshow(pred_raw, vmin=0, vmax=1)
    plt.title('soft prediction')
    plt.subplot(1, 5, 5)
    plt.imshow(true_label)
    plt.title('GT label')
    plt.tight_layout()
    name_string = 'pred_' + data_mode + '.png'
    fname = os.path.join(out_dir, name_string)
    plt.savefig(fname, bbox_inches=0)
    plt.close()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(inp_dem)
    plt.title('DEM')
    plt.subplot(1, 3, 2)
    plt.imshow(pred)
    plt.title('t=0.5')
    plt.subplot(1, 3, 3)
    plt.imshow(true_label)
    plt.title('GT')
    plt.tight_layout()
    name_string = 'pred_binary_' + data_mode + '.png'
    fname = os.path.join(out_dir, name_string)
    plt.savefig(fname, bbox_inches=0)
    plt.close()
    """ PR and ROC Curves """
    print('\nComputing PR and ROC curves')
    y_true = true_label.view(-1).detach().cpu().numpy()
    y_scores = pred_raw.view(-1).detach().cpu().numpy()

    # PR Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    # AUC
    # source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
    auc = roc_auc_score(y_true, y_scores)

    print('average_precision: ', average_precision)
    print('AUC: ', auc)

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision)
    plt.xlim((0.0, 1.0))
    plt.ylim((0, 1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    name_string = 'PR_' + data_mode + '.png'
    fname = os.path.join(out_dir, name_string)
    plt.savefig(fname)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.xlim((0.0, 1.0))
    plt.ylim((0, 1))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUC: ' + str(auc))
    name_string = 'ROC_curve' + data_mode + '.png'
    fname = os.path.join(out_dir, name_string)
    plt.savefig(fname)

    # saving test set metrics
    name_string = 'results_' + data_mode + '.txt'
    fname = os.path.join(out_dir, name_string)
    with open(os.path.join(fname), 'w') as result_file:
        result_file.write('AUC ')
        result_file.write(str(auc))
        result_file.write('\nAverage Precision')
        result_file.write(str(average_precision))

        result_file.write('\nResults on the test set:')
        result_file.write('\nPixel acc ')
        result_file.write(str(pixel_acc))
        result_file.write('\nMean Acc  ')
        result_file.write(str(mean_acc))
        result_file.write('\nIoU full ')
        result_file.write(str(IoU_array))
        result_file.write('\nmean Iou ')
        result_file.write(str(mean_IoU))

    print('\nAll done, results saved in directory: ', out_dir)


if __name__ == '__main__':
    evaluate()
