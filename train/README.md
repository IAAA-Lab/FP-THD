## 1. Installation

### 1.1. Environment

```bash
conda env create -f environment.yml
conda activate htr
```

The code was tested on Python 3.9 and PyTorch 1.13.0.


### 1.2. Datasets

* Using **MDM, Benthman and Rodrigo** for handwritten text recognition.

</summary>
  <details>
   <summary>
   MDM
   </summary>
    
    Send an email to : hala.neji@unizar.es

  </details>
  <details>
   <summary>
   Benthman
   </summary>
    
    Download the dataset from here: https://zenodo.org/records/44519
    
  </details>
  <details>
   <summary>
   Rodrigo
   </summary>
    
    Download the dataset from here: https://zenodo.org/records/1490009
  </details>
  
* Download datasets to ./data/.
Take Molino for an example:
The structure of the file should be:

```
./data/MDM/
├── train.ln
├── val.ln
├── test.ln
└── lines
      ├──Miguel_del_Molino_1585_latin_antiguo_page_659_page_r007-l117.jpeg
      ├──Miguel_del_Molino_1585_latin_antiguo_page_659_page_r007-l117.txt
      ├──Miguel_del_Molino_1585_latin_antiguo_page_659_page_r009-l035.jpeg
      ├──Miguel_del_Molino_1585_latin_antiguo_page_659_page_r009-l035.txt
      ...
```


## 2. Prepare data
In case the line images are not already prepared, this script can be used to generate both the line images and the corresponding **label.pkl** file needed for training.

```
bash

cd data
python  format_MDM_correct_train.py

```

## 3. Train model

* You can use the commands in ./run/ to train and test on different datasets and reproduce the results.


