# MPRe: Multi-scale Feature Guided Prototype Reconstruction for Few-Shot Fine-Grained Image Classification

## Code environment

* All the requirements to run the code are in requirements.txt. You can download requirements by running below script.

  ```shell
  pip install -r requirements.txt
  ```

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN), but the categories  of train, val, and test follows split.txt. And then move the processed dataset  to directory ./data.

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- cars \[[Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)\]
- dogs \[[Download Link](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link)\]

## Train

* To train MPRe on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/MPRe/Conv-4
  ./train.sh
  ```

* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/MPRe/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/BiFRN/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/BiFRN/ResNet-12
    python ./test.py
```
