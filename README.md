# vehicle-detector
This code implements SVM classifier and sliding window technique to detect vehicles in a video. The classifier was trained using HOG features and [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/) datasets.


## Table of contents

- [Explanation](#explanation)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Running the code](#running-the-code)
- [Authors](#authors)

## Explanation


## Prerequisites

You need to install:
- [Python3](https://www.python.org/downloads/)

- [Jupyter Notebook](http://jupyter.org/install/)
    1. `python3 -m pip install -U pip`  
    2.  `python3 -m pip install jupyter`

- [scikit-learn](http://scikit-learn.org/stable/install.html) for preprocessing
    1.  `python3 -m pip install -U scikit-learn`

- [scikit-image](https://scikit-image.org/download.html) for feature extraction
    1.  `python3 -m pip install -U scikit-image`

- **plot** the results by [installing Matplotlib](https://matplotlib.org/users/installing.html) - Linux, macOS and Windows:
    1.  `python3 -m pip install -U matplotlib`
-  show **video** by installing [moviepy](https://zulko.github.io/moviepy/install.html):
    1. `python3 -m pip install -U moviepy`

## Quick-start
To start using the vehicle-detector you need to clone the repo:

```
git clone https://github.com/dweeptrivedi/vehicle-detector.git
```

## Running the code

Training:

  1. car images should be at dataset/vehicles/
  2. noncar images should be at dataset/non-vehicles/
  3. Run the code:
         ```
         jupyter nbconvert --execute train.ipynb --ExecutePreprocessor.timeout=-1
         ```
         
  OR
         
  run `train.ipynb` from jupyter notebook after modifying dataset location

Detection:

run `detect.ipynb` from jupyter notebook after modifying test video/image location

## Authors:
* **Dweep Trivedi** - Please give me your feedback: dweeptrivedi1994@gmail.com

    Feel free to contribute
