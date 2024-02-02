from setuptools import setup, find_packages

setup(
    name="imdeid",
    packages=find_packages(".", exclude=["dicom2nifti"]),
    description="Package for performing Duke Image De-Identification",
    version="1.1.1",
    url="https://github.com/apoorvasikka1/Duke_ImageDeID.git",
    author="Apoorva",
    author_email="apoorva.sikka@nference.net",
    keywords=[
        "pip",
        "ImageDeID",
        "TextNoTextClassifier",
        "HeadNoHeadClassifier",
        "SkullStripping",
        "TextRedaction",
    ],
    install_requires=[
        "opencv-python-headless>=4.8",
        "patchify==0.2.3",
        "Pillow==9.2.0",
        "pydicom==2.3.0",
        "scikit_learn==1.1.2",
        "scipy==1.7.3",
        "tqdm==4.62.3",
        "python-gdcm==3.0.19",
 
       
    ],
    extras_require={
        "testing": ["matplotlib==3.4.2"],
        "gpu": ["cupy-cuda12x==12.0.0"],
    },
    dependency_links=["https://repo.nferx.com/repository/pypi-all/simple/"],
)
