from imdeid.DigiPathDeID.src.data_utils import get_image_data, construct_tile_image
from pydicom import dcmread
from pydicom.encaps import decode_data_sequence
from math import ceil
import numpy as np
from pydicom.encaps import encapsulate
from io import BytesIO
from PIL import Image

#find frame index from a given x,y
def find_frame_index(dicom,x,y):
    # print("x & y",(x,y))
    col_px = dicom[0x0048, 0x0006].value
    # print("col_px",col_px)
    row_px = dicom[0x0048, 0x0007].value
    # print('row_px',row_px)
    tile_height = dicom[0x0028, 0x0010].value
    # print('tile_height',tile_height)
    tile_width = dicom[0x0028, 0x0011].value
    # print('tile_width',tile_width)
    # frame_count = dicom[0x0028, 0x0008].value
    # print("Number of frames: ", frame_count)
    frame_indx = (ceil(y/tile_height)-1)*ceil(col_px/tile_width) + ceil(x/tile_width) -1
    return frame_indx

#write a frame on disk given frame_indx
def write_frame(dicom,frame_indx,save_name):
    pixels = dicom[0x7fe0, 0x0010].value
    frames = decode_data_sequence(pixels)
    p=frames[frame_indx] 
    with open(save_name, 'wb') as f: 
        f.write(p)
        
#write a frame on the disk given x,y
def save_frame_from_xy(dicom,x,y,save_name):
    frame_indx = find_frame_index(dicom,x,y)
    write_frame(dicom,frame_indx,save_name)

    
def find_frame_indices(orig_dicom_ds,new_x1,new_y1,new_x2,new_y2):
    frames = []
    for y_ in range(new_y1,new_y2):
        f1_s = find_frame_index(orig_dicom_ds,new_x1,y_)
        f1_e = find_frame_index(orig_dicom_ds,new_x2,y_)
        frames.extend(list(range(f1_s,f1_e+1)))
    frames = np.unique(frames)
    return frames
    
def redact_a_tile_with_white_for_a_xy_and_save_new_dicom(dicom_path,indices,new_dicom_path):
    #read the dicom
    ds = dicom.dcmread(dicom_path)
    
    #define white tile
    white_image = Image.fromarray(255*np.ones((256,256,3), dtype=np.uint8))
    buffered = BytesIO()
    white_image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    
    #get frame bytes stream
    pixels = ds[0x7fe0, 0x0010].value
    frames = decode_data_sequence(pixels)
    #overwrite the particular tile with white
    for idx in indices:
        frames[idx] = image_bytes
    #encapsulate the byte stream
    ds[0x7fe0, 0x0010].value = encapsulate(frames, has_bot=False)
    #save the new dicom
    ds.save_as(new_dicom_path)
    return ds