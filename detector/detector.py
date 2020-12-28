from functools import partial
import os
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt

import torch
from PIL import Image
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


from numpy import moveaxis
from numpy import asarray
from PIL import Image

from dataset import PeopleDataset
import transforms as T
from engine import train_one_epoch, evaluate
import utils


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def plot_detections(image, detections):

    # load the color image
    data_ = asarray(image)
    # change channels last to channels first format
    data_ = moveaxis(data_, 2, 0)
    data_ = moveaxis(data_, 2, 0)


    img=data_
    plt.figure(figsize=(10,10))
    #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_image = img
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # #detections = y.data
    # # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(0)):
        detection = detections[i]
        pt = detection.cpu().detach().numpy()
        tl = (pt[0], pt[1])
        width = pt[2]-pt[0]
        height = pt[3]-pt[1]
        coords = tl, width, height
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], 'detected_person', bbox={'facecolor':color, 'alpha':0.5})


def get_fasterrcnn_model():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_mobilenet_backbone_model():

    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def train(model):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # use our dataset and defined transformations
    dataset = PeopleDataset(None)
    train_dataset, test_dataset = dataset.train_test_split(0.8)
    train_dataset.transforms = get_transforms(train=True)
    test_dataset.transforms = get_transforms(train=False)


    # split the dataset in train and test set
#     indices = torch.randperm(len(dataset)).tolist()
#     dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
#     dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


def test_model_and_data_loader():

    model = get_fasterrcnn_model()
    dataset = PeopleDataset(get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)
    # For Training
    # images, targets = next(iter(data_loader))
    # images = list(image for image in images)
    # targets = [{k: v for k, v in t.items()} for t in targets]
    # output = model(images,targets)   # Returns losses and detections
    # # For inference
    # model.eval()

    images, targets = next(iter(data_loader))
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    prediction = predictions[0]
    detections = prediction['boxes'][prediction['scores']>0.7]
    plot_detections(images[0], detections)


def make_detections_on_test_set(model, dataset_test, sequences_to_evaluate):

    for dir_name in sequences_to_evaluate:

        dataset_test.data_df = dataset_test.data_df.loc[dataset_test.data_df['dir_name']==dir_name]
        data_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=4, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        all_frame_results_array = None
        images_processed = 0
        data_loader = iter(data_loader)
        while True:

            try:
                images, targets = next(data_loader)
            except StopIteration:
                break

            images_processed += len(images)
            print('images_processed: ', str(images_processed))

            with torch.no_grad():
                predictions = model(images)

            for frame, target in zip(predictions, targets):
                boxes = frame['boxes'].cpu().numpy()
                labels = frame['labels'].cpu().numpy().reshape(-1,1)
                scores = frame['scores'].cpu().numpy().reshape(-1,1)
                image_id = np.array([int(os.path.splitext(target['image_id'])[0])]*scores.shape[0]).reshape(-1,1)
                x = np.array([-1]*scores.shape[0]).reshape(-1,1)
                y = np.array([-1]*scores.shape[0]).reshape(-1,1)
                z = np.array([-1]*scores.shape[0]).reshape(-1,1)

                # Convert bbox coords to tlwh from tlbr
                boxes[:,2] = boxes[:,2]-boxes[:,0]
                boxes[:,3] = boxes[:,3]-boxes[:,1]


                frame_results = np.hstack([image_id, labels, boxes, scores, x, y, z])
                if all_frame_results_array is not None:
                    all_frame_results_array = np.vstack([all_frame_results_array, frame_results])
                else:
                    all_frame_results_array = frame_results

        os.makedirs(os.path.join('data', dir_name, 'det'), exist_ok=True)
        np.savetxt(os.path.join('data', dir_name, 'det', 'det.txt'), all_frame_results_array, delimiter=",")
