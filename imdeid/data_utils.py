import cv2
import imgproc
import torch

import numpy as np
import pandas as pd

import pydicom as dicom
from pydicom.dataset import Dataset
import pydicom.pixel_data_handlers.util as util

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from utils import Box
import concurrent.futures
from io import BytesIO
from PIL import Image
from itertools import starmap
from pydicom.encaps import decode_data_sequence

import multiprocessing
num_processes = multiprocessing.cpu_count()-8


@dataclass
class DicomProcessingResult:
    keep_dicom: bool
    dicom_SOPID: str
    PhotometricInterpretation: Optional[str] = None
    PixelArrayLoaded: bool = False
    boxes: Dict[int, List[Box]] = field(default_factory=dict)


def construct_tile_image(img, rows, cols, frame_height, frame_width, totalPixelMatrixRows,totalPixelMatrixColumns):
    tiled_image = np.zeros((rows*frame_height,cols*frame_width,3), dtype=np.uint8) 
    for i in range(rows):
        for j in range(cols):
            frame = img[i * cols + j]
            x_offset = j * frame_width
            y_offset = i * frame_height
            tiled_image[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width, :] = frame
    tiled_image = tiled_image[0:totalPixelMatrixRows,0:totalPixelMatrixColumns]
    return tiled_image

def get_image_data(ds):
    try:
        images = ds.pixel_array
        frame_height = images.shape[1]
        frame_width = images.shape[2]
        rows = int(np.ceil(ds.TotalPixelMatrixRows/images.shape[1]))
        cols = int(np.ceil(ds.TotalPixelMatrixColumns/images.shape[2]))
        images = construct_tile_image(images, rows, cols, frame_height, frame_width, ds.TotalPixelMatrixRows, ds.TotalPixelMatrixColumns)
        if ds.PhotometricInterpretation == 'YBR_FULL_422':
            images = util.convert_color_space(images,'YBR_FULL_422','RGB')
    except Exception:
        images = None
    return images


class ImageDataset(Dataset):
    def __init__(self, headers):
        self.headers = headers

    def __getitem__(self, index): 
        try:
            header = self.headers[index]
            ds_head = dicom.dcmread(header.dicom_SOPID)
            #images = ds_head.pixel_array
            images = convert_to_1x(ds_head) 
            #if ds_head.PhotometricInterpretation == 'YBR_FULL_422':
            #    images = util.convert_color_space(images,'YBR_FULL_422','RGB')
            cv2.imwrite('/root/apoorva/jpegs/'+header.dicom_SOPID.split('/')[-1].replace('.dcm','.jpeg'),images)
            
        except Exception:
            return None  # Move to the next index  
        return images, index

    def __len__(self):
        return len(self.headers)


def collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        # If all samples are None, return None
        return torch.empty(0), torch.empty(0)
    else:
        images, indexes = [], []
        image_, index_ = zip(*batch)
        # Split the 4D image into individual frames
        for i in range(len(image_)):
            image = image_[i]
            index = index_[i]
            images.append(image)
            indexes.append(index)
        return images, indexes

def format_headers(dicom_headers):
    rows_list,columns_list,frames_list,imagetype_list,SOP_id_list,photometric_rep_list=[],[],[],[],[],[]
    for dicom_header in dicom_headers:
        rows_list.append(dicom_header.Rows)
        columns_list.append(dicom_header.Columns)
        frames_list.append(dicom_header.NumberOfFrames)
        imagetype_list.append(dicom_header.ImageType)
        SOP_id_list.append(dicom_header.SOPInstanceUID)
        photometric_rep_list.append(dicom_header.PhotometricInterpretation)
    

    # Assuming you have lists like min_rows, min_columns, and min_frames
    data = {
        "Imagetype": imagetype_list,
        "Rows": rows_list,
        "Columns": columns_list,
        "Numframes": frames_list,
        "SOP Instance UID": SOP_id_list,
        "PhotometricInterpretation": photometric_rep_list
    }

    df = pd.DataFrame(data)
    return df

def contains_values(imagetypes, values):
    return all(value in imagetypes for value in values)
 
def find_onex_dicom(df):
    values_to_check = ['DERIVED', 'RESAMPLED']
    df_s = df[df['Imagetype'].apply(lambda x: contains_values(x, values_to_check))]
    df_sub = df_s[(df_s['Rows'] == 256) & (df_s['Columns']==256)]
    if len(df_s)>0: #if derived and resampled images exists, then I look for smaller magnifciation
        df_sub2 = df_sub[(df_sub['Numframes']>=6) & (df_sub['Numframes']<=30)]
        if len(df_sub2)>0:
            return df_sub2.iloc[0]['SOP Instance UID'], False
        else:
            min_value = df_sub['Numframes'].min()
            df_sub3 = df_sub[df_sub['Numframes'] == min_value]
            return df_sub3.iloc[0]['SOP Instance UID'], True #smallest possible compression value
    else: #else I return 40x image SOP UID
        values_to_check = ['ORIGINAL', 'VOLUME']
        df_s = df[df['Imagetype'].apply(lambda x: contains_values(x, values_to_check))]
        return df_s.iloc[0]['SOP Instance UID'], True
     
    
def run_digipath_deid(ip_paths,base_dir) -> List[DicomProcessingResult]:
    """
    Args:
        List[ds (DICOM obj)]: List of DICOM object of the instance loaded using pydicom
    Returns:
        DicomProcessingResult: A DicomProcessingResult instance containing SOP IDs and boolean variables.
    """
    digipath_headers=[]
    for each in ip_paths:
        result = DicomProcessingResult(keep_dicom=False, dicom_SOPID=base_dir+each)
        digipath_headers.append(result)
    return digipath_headers

    
def calculate_final_image_size(image_size, grid_cols_, grid_rows_, max_frames):
    num_frames = image_size[0]
    # Check if the number of frames is less than or equal to the maximum allowed frames
    if num_frames <= max_frames:
        return image_size, grid_cols_, grid_rows_

    # Define the downsampled dimensions (256x256 tiles downsampled to 6x256x256)
    tile_height = 256
    tile_width = 256
    downsampled_image_size = (
        image_size[0],
        tile_height // 2,
        tile_width // 2,
        image_size[3],
    )

    # Calculate the grid dimensions
    grid_cols = np.ceil(grid_cols_ / tile_width)
    grid_rows = np.ceil(grid_rows_ / tile_height)

    # Calculate the new grid dimensions
    new_grid_pixelcols = int(downsampled_image_size[1] * (grid_cols_ / 256))
    new_grid_pixelrows = int(downsampled_image_size[2] * (grid_rows_ / 256))

    # Calculate the number of frames with the downsampled image size
    final_num_frames = np.ceil(new_grid_pixelcols / 256) * np.ceil(
        new_grid_pixelrows / 256
    )
    # print(final_num_frames,new_grid_pixelcols, new_grid_pixelrows )
    downsampled_image_size = (
        final_num_frames,
        image_size[1] // 2,
        image_size[2] // 2,
        image_size[3],
    )
    # Check if the number of frames with downsampled image size is less than or equal to the maximum allowed frames
    if final_num_frames <= max_frames:
        return image_size, grid_cols_, grid_rows_

    # Continue to calculate the final image size recursively
    return calculate_final_image_size(
        downsampled_image_size, new_grid_pixelcols, new_grid_pixelrows, max_frames
    )



def convert_frames(frame_number,frame_bytes,tile_size):
    frame_image = (np.array(Image.open(BytesIO(frame_bytes))))
    resized_tile = cv2.resize(
            frame_image, tile_size, interpolation=cv2.INTER_AREA
        )
    return frame_number, resized_tile


def resize_and_stack_all_tiles(ds, num_tiles, tile_size, grid_cols_, grid_rows_):
    resized_tiles = []
    pixels = ds[0X7FE0,0X0010].value
    frames = decode_data_sequence(pixels)
    frames_data = [(i,frames[i],tile_size) for i in range(len(frames))]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(pool.starmap(convert_frames, frames_data))

    resized_tiles = [frame_data for _,frame_data in results]

    resized_image = np.array(resized_tiles)
    grid_cols = np.ceil(ds.TotalPixelMatrixColumns / 256)
    grid_rows = np.ceil(ds.TotalPixelMatrixRows / 256)
    
    # Create an empty big grid image
    im = im = construct_tile_image(
        resized_image, int(grid_rows), int(grid_cols), tile_size[0], tile_size[1],grid_rows_, grid_cols_
    )
    return im


def convert_to_1x(ds):
    original_num_frames = ds.NumberOfFrames
    image_size = (original_num_frames, 256, 256, 3)
    max_frames = 4

    final_image_size, grid_cols, grid_rows = calculate_final_image_size(
        image_size, ds.TotalPixelMatrixColumns, ds.TotalPixelMatrixRows, max_frames
    )
    
    resized_image = resize_and_stack_all_tiles(
        ds,
        original_num_frames,
        (final_image_size[1], final_image_size[2]),
        grid_cols,
        grid_rows,
    )
    #if ds.PhotometricInterpretation == "YBR_FULL_422":
    #    resized_image = util.convert_color_space(
    #        resized_image, "YBR_FULL_422", "RGB"
    #    )
    return resized_image
    
