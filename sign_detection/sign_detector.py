import os
import numpy as np
import cv2
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from engine import train_one_epoch, evaluate
from PIL import Image, ImageDraw #, ImageFont
from pycocotools.coco import COCO

from utils_custom import get_transform, collate_fn_custom, get_model_instance_detection
from text_recognizer import recognize_text


data_dir = 'data/images'
coco_dir = 'coco/output.json'


model_output = 'model_output/fuel_sign_model.pt' # 'model_output/fuel_detector_2.pt'
prediction_save_path = 'prediction_output'

east_text_path = 'frozen_east_text_detection.pb'

test_fraction = 0.5


class fuelDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotations, transforms=None):
        self.root = root
        self.transforms = transforms
        
        # self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))

        self.coco = COCO(annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):

        # Coco annotations
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        coco_annotation = coco.loadAnns(ann_ids)

        # Load image
        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        num_objs = len(coco_annotation)

        # Bounding boxes
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels - only one label in this case
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Tensorize image id
        img_id = torch.tensor([img_id])

        # Size of bounding box
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Target dictionary
        target = {}

        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = img_id
        target['area'] = areas
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
            # img = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def train():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Our dataset has two classes only - background and fuel station sign
    num_classes = 2

    dataset = fuelDataset(
        # root='data/images', annotations='coco/output.json', transforms=get_transform(train=False)
        root=data_dir, annotations=coco_dir, transforms=get_transform(train=False)
    )
    dataset_test = fuelDataset(
        # root='data/images', annotations='coco/output.json', transforms=get_transform(train=False)
        root=data_dir, annotations=coco_dir, transforms=get_transform(train=False)
    )

    # Size defining training - testing split
    N_split = int(len(dataset) - np.floor(len(dataset) * test_fraction))

    print('Split index: ', N_split)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-5])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])
    dataset = torch.utils.data.Subset(dataset, indices[:N_split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[N_split:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn_custom
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_custom
    )

    # Retreive model
    model = get_model_instance_detection(num_classes)

    # Move model to the right device - aka. mount CUDA gpu if available
    model.to(device)

    # Construnction of optimzier and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 3

    # Training loop
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Evaluate model on test set
        evaluate(model, data_loader_test, device=device)

    print(' \n ######## Finished training! ########')

    # torch.save(model, 'model_output/fuel_detector_2.pt')
    torch.save(model, model_output)

    print(f'Has saved model at {model_output}')


def detect_fuel_station(image_path):
    """
    Detect fuel station image on trained model

    Args:
        image file path
    Returns:
        [x1, y1, x2, y2], label (tuple of list (of floats) and string)

    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = torch.load(model_output)

    print(f'Loading model from {model_output}')

    # Neccessary in order to initialize all batch normalizations and ...
    model.eval()

    # Open image to infer
    img = Image.open(image_path).convert('RGB')
    # orig_img = img.copy()

    # Extract and apply transforms in order to match model image format
    transformations = get_transform(train=False)
    img = transformations(img)

    img = img.to(device)
    img = img.unsqueeze(1).float()

    # Run forward pass to detect and extract predicted bounding boxes
    with torch.no_grad():
        detections = model(img)

    # TODO: implement non-maximum supression of infered predictions ???



    box, label = None, None
    
    try:
        detection = detections[0]
        box = detection['boxes'].tolist()[0]
        label = int(detection['labels'].tolist()[0])
    except:
        print(f'No valid predictions for image from {image_path}')
    finally:
        pass
    return (box, label)


def draw_prediction(box, image_path, save_path=None, save_name=None):
    """
    Draws predicted bounding box on image.
    
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = box

    start = (int(x1), int(y1))
    stop = (int(x2), int(y2))
    xy = [start, stop]

    draw.rectangle(xy, outline='aqua', width=5) # outline='red'
    del draw

    if save_path and save_name:
        print(f'Saving {save_name} to {save_path}...')
        img.save(os.path.join(save_path, save_name))
    else:
        print(f'Provide valid save path folder and file name with ending (ex: .png).')

    img.show()


def detect_fuel_price_from_image(image_path):
    """
    Detects fuel price.
    
    Returns:
        string denoting price.
        TODO: Process price result, maybe check if int/float...

    """

    box, _ = detect_fuel_station(image_path)

    if not box:
        return None

    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    image = cv2.imread(image_path)

    image_cropped = image[y1:y2, x1:x2]
    # cv2.imwrite('crop_boi_3.png', image_cropped)

    # cv2.imshow("CROP! ", image_cropped)
    # cv2.waitKey(0)

    # image_u = cv2.imread('crop_boi_3.png')

    price_result = recognize_text(image_cropped, east_text_path)

    return price_result


if __name__ == "__main__":

    # NOTE: UNCOMMENT train() IF TRAINING NEW NETWORK
    # CHANGE global variable 'model_output_dir' TO WRITE TO NEW MODEL
    
    # train()


    """

    ### EXAMPLE INFERENCE ###
    example_image = 'data/images/2.png'
    example_save_name = '2_test_1.png'

    print(f'Prediction location of sign in {example_image}. \n')

    box, label = detect_fuel_station(example_image)
    draw_prediction(box, example_image, prediction_save_path, example_save_name)

    img = Image.open(example_image).convert('RGB')

    x1, y1, x2, y2 = box

    img_cropped = img.crop((x1, y1, x2, y2))
    img_cropped.save('crop_boi_2.png')

    # img_cropped.show()

    # recognize_text('crop_boi.png', 'frozen_east_detection.pb')

    ### END OF EXAMPLE ###

    """

    price_result = detect_fuel_price_from_image('data/13.png')

    print('Fuel price: ', price_result)
