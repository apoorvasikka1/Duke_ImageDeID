# De-identification System for Digital Pathology Images
Overview
The De-identification System for Digital Pathology Images is a software solution designed to remove the images with sensitive patient information. Digital pathology involves the analyzing and detecting text of high-resolution pathological images and further doinf FPR reduction.

## Requirements

- Python 3.10 or higher

## Installation

1. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```python
# Example code snippet
python3 run_digipath_pipeline.py --cuda_bool=False --ip_dir '/home/overview_images/' --op_dir '/home/sample_op'

