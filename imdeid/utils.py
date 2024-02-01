import os
import cv2
import csv
import math
import numpy as np

import torch
from torch.autograd import Variable
from dataclasses import dataclass, asdict

import imgproc
import craft_utils
from collections import OrderedDict

class Box:
    """
    Represents a box with its dimensions and associated properties.

    Attributes:
        id (int): The unique identifier of the box.
        y (float): The y-coordinate of the box.
        x (float): The x-coordinate of the box.
        h (float): The height of the box.
        w (float): The width of the box.
        NearestBlocKMerge (bool): Flag indicating if the box is merged with the nearest block.
        text (str): The text associated with the box.
        retainbox (bool): Flag indicating if the box should be retained.

    Methods:
        get_dimensions: Returns a tuple of the box's dimensions.

    Raises:
        None.
    """

    def __init__(self, id, box_, BarcodeBox):
        """
        Initializes a Box object.

        Args:
            id (int): The unique identifier of the box.
            box_ (tuple): A tuple containing the x, y, width, and height values of the box.
            NearestBlocK (bool, optional): Flag indicating if the box is merged with the nearest block. Defaults to False.

        Returns:
            None.

        Raises:
            None.
        """
        self.id = id
        self.y = box_[0]
        self.x = box_[1]
        self.h = box_[2]
        self.w = box_[3]
        self.BarcodeBox = BarcodeBox
        self.text = ''
        self.retainbox = False

    def get_dimensions(self):
        return (self.id, self.y, self.x, self.h, self.w)

    def to_dict(self):
        """
        Returns the dictionary representation of the Box object.

        Returns:
            dict: Dictionary representing the Box object.
        """
        return {
            "id": self.id,
            "y": self.y,
            "x": self.x,
            "h": self.h,
            "w": self.w,
            "BarcodeBox": self.BarcodeBox,
            "text": self.text,
            "retainbox": self.retainbox,
        }

    @classmethod
    def from_dict(cls, data):
        """
        Creates a Box object from a dictionary.

        Args:
            data (dict): Dictionary representing the Box object.

        Returns:
            Box: Box object created from the dictionary.
        """
        obj = cls(
            id=data["id"],
            box_=(data["y"], data["x"], data["h"], data["w"]),
            NearestBlocK=data["NearestBlocKMerge"],
        )
        obj.text = data["text"]
        obj.retainbox = data["retainbox"]
        return obj

def clamp_coordinates(box,img_shape):
    box_new = []
    for each in box:
        if (each[0]<0):
            each[0] = 0
        elif (each[1]<0):
            each[1] = 0
        elif (each[0])>=img_shape[1]:
            each[0] = img_shape[1]
        elif (each[1])>=img_shape[0]:
            each[1] = img_shape[0]
        box_new.append(each)
    return box_new 

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    #used for craft
    boxes_,pixel_values_list=[],[]
    for each in image:
        canvas_size = 1280
        mag_ratio = 1.5
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(each, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        bboxes_barcode = detect_barcode(each)
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

        if cuda:
            x = x.cuda()
            
        # forward pass
        with torch.no_grad():
            y, feature = net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if refine_net is not None:
            with torch.no_grad():
                y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
        
        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
   
        boxes = convert_boxes(boxes,each.shape)
        boxes_.append(boxes+bboxes_barcode)
        pixel_values_list.extend(crop_image_regions(each, boxes))
        pixel_values_list.extend(crop_image_regions(each, bboxes_barcode))
    return boxes_, pixel_values_list


def crop_image_regions(image, m_bboxes, target_size=(384,384)):
    """
    Crop image regions based on the provided bounding boxes.

    Args:
        image_frame (numpy.ndarray): The image frame from which to crop the regions.
        m_bboxes (list): A list of bounding boxes.
        
    Returns:
        list: A list of cropped image regions.

    Raises:
        None.
    """
    crops_list = []
    pixel_values_list = []
    for box in m_bboxes:
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h
        crop_arr = image[int(x1) : int(x2), int(y1) : int(y2)]
        im_crop = cv2.resize(crop_arr, target_size)  # Resize using cv2.resize
        im_crop = torch.from_numpy(im_crop.transpose((2, 0, 1)))
        im_crop = im_crop.to(torch.float32) 
        im_crop = im_crop.unsqueeze(0)
        pixel_values_list.append(im_crop)
    return pixel_values_list

def convert_boxes(bboxes, im_shape, BarcodeText=False):
    """Created cropped images from list of bounding boxes
    Args:
        img: input image
        boxes: bounding boxes of shape (N, 4) where N is the number of boxes, and the relative
            coordinates (xmin, ymin, xmax, ymax)
        channels_last: whether the channel dimensions is the last one instead of the last one
    Returns:
        list of cropped images
    """
    m_bboxes = []
    for index, box in enumerate(bboxes):
        box = clamp_coordinates(box,im_shape)
        a1 = min(box[0][1],box[1][1],box[2][1],box[3][1])
        a2 = max(box[0][1],box[1][1],box[2][1],box[3][1])
        a3 = min(box[0][0],box[1][0],box[2][0],box[3][0])
        a4 = max(box[0][0],box[1][0],box[2][0],box[3][0])

        if a3==a4:
            a4= a4+1
            
        x1, x2, y1, y2 = a1,a2,a3,a4
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        box_ = y, x, h, w
        m_bboxes.append(Box(index, box_, BarcodeText))
    return m_bboxes

def assign_recognized_texts(
    bboxes_, rec_text_list, header_indexes, headers
):
    """
    Merge recognized texts based on the merged bounding boxes.

    Args:
        rec_text_list (list): A list of recognized texts for each bounding box.
        merged_boxes_ref (list): A list of merged bounding boxes.

    Returns:
        tuple: A tuple containing the updated merged bounding boxes and the final list of merged texts.

    Raises:
        None.
    """
    count=0
    for i, each_frame in enumerate(bboxes_):
        header_index = header_indexes[i]
        
        frame_boxes, box_keys, headers_, filenames_ = [], [], [], []
        for j, sub_box in enumerate(each_frame):
            sub_box.text = 1
            if (rec_text_list[count]==0) & (sub_box.BarcodeBox == False):
                sub_box.retainbox = True
            frame_boxes.append(sub_box)
            count += 1
        headers[header_index].boxes = frame_boxes  # Extend the nested dictionary
    return headers

def find_contours(gray_image):
    # Define the kernel for dilation and erosion
    kernel = np.ones((15, 15), np.uint8)  # You can adjust the kernel size
    dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
    closed_image = cv2.erode(dilated_image, kernel, iterations=1)
    _, binary_mask = cv2.threshold(closed_image, 100, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_mask)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #filter them by size 
    min_contour_area = 1000  # Adjust this threshold as needed
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
    
    return filtered_contours

def find_rect_contours(contours, image_shape):
    rectangular_contours=[]
    #filter them by resoltion
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx1=[]
        for each in approx:
            approx1.append(each[0])
        approx1 = np.array(approx1)
        width = max(approx1[:, 0]) - min(approx1[:, 0])
        length = max(approx1[:, 1]) - min(approx1[:, 1])

        if len(approx1)>5:
            continue
        elif ((width/image_shape[1]) > 0.8) and ((length/image_shape[0]) > 0.8):
            continue
        elif ((width/image_shape[1]) > 0.8) or ((length/image_shape[0]) > 0.8):
            rectangular_contours.append(contour)
    return rectangular_contours

def detect_barcode(image):
    bboxes_=[]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_contours = find_contours(gray_image)
    rectangular_contours = find_rect_contours(filtered_contours, gray_image.shape)
    
    # Get bounding box coordinates for each contour
    bounding_boxes = [cv2.boundingRect(contour) for contour in rectangular_contours]

    # Display bounding box coordinates
    for index, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        box_ = x, y, w, h
        bboxes_.append(Box(index, box_, True))
    return bboxes_


def compute_downsampling_level(orig_dicom_ds,onex_dicom_ds):
    
    rows_downsampling_factor = orig_dicom_ds.TotalPixelMatrixRows / onex_dicom_ds.TotalPixelMatrixRows
    columns_downsampling_factor = orig_dicom_ds.TotalPixelMatrixColumns / onex_dicom_ds.TotalPixelMatrixColumns
    result_rows = int(math.log(rows_downsampling_factor, 2))
    result_cols = int(math.log(columns_downsampling_factor, 2))
    assert result_rows == result_cols
    return rows_downsampling_factor

def compute_newbox_coordinates(x1,y1,x2,y2,downsample_factor):
    new_x1,new_y1 = x1*int(downsample_factor), y1*int(downsample_factor)
    new_x2,new_y2 = x2*int(downsample_factor), y2*int(downsample_factor)
    return new_x1,new_y1, new_x2,new_y2
    
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def update_box_coordinates_40x(digipath_headers, index, downsample_level):
    updated_boxes = []
    for each in digipath_headers[index].boxes:
        new_x1, new_y1, new_x2,new_y2 = compute_newbox_coordinates(each.x, each.y,each.x+each.w,each.y+each.h, downsample_level)
        each.x = new_x1
        each.y = new_y1
        each.w = new_x2-new_x1
        each.h = new_y2-new_y1
        updated_boxes.append(each)
    digipath_headers[index].boxes = updated_boxes
    return digipath_headers

def save_headers_ascsv(updated_headers, out_dir):
    create_directory(out_dir)
    # Convert each dataclass instance to a dictionary
    dict_list = [asdict(instance) for instance in updated_headers]
    # Extract the dictionary and remove unwanted keys
    dict_list = [{k: v for k, v in item.items() if k not in ['boxes', 'PhotometricInterpretation','PixelArrayLoaded']} for item in dict_list]


    # CSV file path
    file_exists=False
    file_path = out_dir + 'output_noclass.csv'

    try:
        with open(file_path,'r') as file:
            headers = next(csv.reader(file))
            file_exists = True
    except FileNotFoundError:
        pass

    # Writing CSV file
    with open(file_path, 'a', newline='') as csv_file:
        # Assuming the first dictionary in the list contains all the keys
        fieldnames = dict_list[0].keys()

        # Create a CSV writer object
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header
        if not file_exists:
            csv_writer.writeheader()

        # Write the data
        csv_writer.writerows(dict_list)

    #print(f"CSV data written to '{file_path}'.")