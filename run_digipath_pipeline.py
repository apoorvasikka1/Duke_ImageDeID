import os
import sys
import cv2
import argparse

import numpy as np
from tqdm import tqdm

import torch 
import torch.nn as nn
from torchvision import models
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pydicom as dicom
import pydicom.pixel_data_handlers.util as util
from typing import List, Optional, Dict, Tuple

from craft import CRAFT
from data_utils import (
    ImageDataset,
    DicomProcessingResult,
    collate_fn,
    run_digipath_deid,
)

from utils import (
    copyStateDict,
    test_net,
    convert_boxes,
    assign_recognized_texts,
    create_directory,
    save_headers_ascsv
)

# FPRClassifier class as provided in your previous messages
class FPRClassifier:
    def __init__(self, cuda_bool, weights_path, num_classes=2):
        self.cuda_bool = cuda_bool
        self.fpr_classifier = models.resnet50(pretrained=False)
        num_ftrs = self.fpr_classifier.fc.in_features
        self.fpr_classifier.fc = nn.Linear(num_ftrs, num_classes)
        checkpoint = torch.load(weights_path,map_location=torch.device('cpu'))
        self.fpr_classifier.load_state_dict(checkpoint)
        if self.cuda_bool:
            self.fpr_classifier = self.fpr_classifier.cuda()
            self.fpr_classifier = torch.nn.DataParallel(self.fpr_classifier)
            cudnn.benchmark = False
        self.fpr_classifier.eval()

    def predict(self, input_data):
        with torch.no_grad():
            outputs = self.fpr_classifier(input_data)
            _, predicted = torch.max(outputs.data, 1)
        return predicted
    
class TextRedaction:
    def __init__(self, craft_model_dir, cuda_bool, device, model_type="craft", fpr_weights_path=None):
        self.model_type = model_type
        self.cuda_bool = cuda_bool
        self.device = device
        if model_type == "craft":
            self.detector = CRAFT()
            #print('Loading weights from checkpoint (' + craft_model_dir + ')')
            if cuda_bool:
                self.detector.load_state_dict(copyStateDict(torch.load(craft_model_dir)))
            else:
                self.detector.load_state_dict(copyStateDict(torch.load(craft_model_dir, map_location='cpu')))

            if self.cuda_bool:
                self.detector = self.detector.cuda()
                self.detector = torch.nn.DataParallel(self.detector)
                cudnn.benchmark = False
            self.detector.eval()
        else:
            print("%s not a valid model", model_type)
            sys.exit()
            
        if fpr_weights_path:
            self.fpr_classifier = FPRClassifier(self.cuda_bool, fpr_weights_path)

    
    def detect_text(self, image):
        link_threshold = 0.4
        text_threshold = 0.7
        low_text = 0.4
        poly_bool = False
        bboxes, pixel_values_list = test_net(self.detector, image, text_threshold, link_threshold, low_text, self.cuda_bool, poly_bool, refine_net=None)   
        return bboxes, pixel_values_list
    
    def run_classifier(
        self, bboxes_, pixel_values_list, image, header_indexes, headers
    ) -> List[DicomProcessingResult]:
        
        predicted_class_ = []
        
        if len(pixel_values_list) > 0:
            '''
            if torch.cat(pixel_values_list).shape[0] < 32:
                pixel_values_combined = torch.cat(pixel_values_list, dim=0)
                
                pred_class = self.fpr_classifier.predict(
                    (pixel_values_combined).to(self.device)
                )
                
                predicted_class_.extend(pred_class)
                # print(generated_ids_)
            else:
                sub_batch_size = 32  # Specify the desired batch size
                pixel_values_combined = torch.cat(pixel_values_list)
                num_batches = (
                    pixel_values_combined.shape[0] + sub_batch_size - 1
                ) // sub_batch_size

                for i in range(num_batches):
                    start_idx = i * sub_batch_size
                    end_idx = min(
                        (i + 1) * sub_batch_size, pixel_values_combined.shape[0]
                    )
                    pixel_values_batch = pixel_values_combined[start_idx:end_idx]
                    pred_class = self.fpr_classifier.predict(
                        pixel_values_batch.to(self.device)
                    )
                    predicted_class_.extend(pred_class)

                if self.device == "cuda":
                    torch.cuda.empty_cache()
            '''
            #print(predicted_class_)
            predicted_class_= [1]*len(pixel_values_list)
            headers = assign_recognized_texts(
                bboxes_, predicted_class_, header_indexes, headers
            )
        
        del pixel_values_list
        return headers
           
        
    def run_textredaction(
        self,
        headers: List[DicomProcessingResult],
        batch_size: int = 16,
    ) -> List:
        
        if len(headers) == 0:
            raise ValueError("No Input paths specified")

        dataset = ImageDataset(headers)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        mini_batch_size = batch_size
        for batch_images, indexes in (data_loader):
            if batch_images == []:
                continue
                
            for index in range(0, len(batch_images), mini_batch_size):
                batch_images_ = batch_images[index : index + mini_batch_size]
                batch_indexes = indexes[index : index + mini_batch_size]
                out, pixel_values_list = self.detect_text(batch_images_)
                headers = self.run_classifier(
                    out, pixel_values_list, batch_images_, batch_indexes, headers
                )
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        return headers

def update_headers(digipath_headers):
    updated_headers = digipath_headers
    for each in range(len(digipath_headers)):
        header = digipath_headers[each]
        text= False
        if len(header.boxes)==0:
            updated_headers[each].keep_dicom = True
        else:
            for each1 in header.boxes:
                if (each1.retainbox == False) or (each1.BarcodeBox ==True):
                    text= True
            if text == False:
                updated_headers[each].keep_dicom = True
    return updated_headers

def create_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Specify Input output directories and device.')

    # Add command line arguments
    parser.add_argument('--ip_dir', type=str, help='Input directory path', required=True)
    parser.add_argument('--op_dir', type=str, help='Output directory path', required=True)
    parser.add_argument('--cuda_bool', type=bool, default = False)
    # Parse the command line arguments
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    '''
     (Input - List of overview images/dir)
    - Reject Image
        - True when overview image is found
        - False when we find a overview image

      - (Input - DICOM of interest - path)  
        - load the dicom and get the pixel array
        - Detect barcode
        - Detect boxes
        - FPR
        -Writes a json corresponding to all image paths whether to discard or keep it.
    '''
    # Get the directory of the script
    args = create_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    craft_model_dir = os.path.join(script_dir,'model_weights/craft_models/craft_mlt_25k.pth')
    fpr_model_dir = os.path.join(script_dir, 'model_weights/tissue_text_models/v139.pth')

    out_dir = args.op_dir
    batch_size = 16
    cuda_bool = args.cuda_bool
    if cuda_bool is True:
        device='cuda'
    else:
        device='cpu'
    ################################################
    #give a list of headers in a series
    base_dir = args.ip_dir
    ip_paths = os.listdir(base_dir)
    headers = run_digipath_deid(ip_paths, base_dir)
    ########################################################
    text_det = TextRedaction(craft_model_dir, cuda_bool, device, "craft",fpr_model_dir)
    digipath_headers = text_det.run_textredaction(headers, batch_size)
    updated_headers = update_headers(digipath_headers) 
    save_headers_ascsv(updated_headers,out_dir)
    
    