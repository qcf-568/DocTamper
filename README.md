# DocTamper
The DocTamper dataset is now avaliable at [BaiduDrive](https://pan.baidu.com/s/1nEEnq1ZWIem7wnkQ1YdTNw?pwd=od9k) and Google Drive ([part1](https://drive.google.com/file/d/150teGvJbtWSULljrh9Sp_NrTlEXKPsTm/view?usp=drive_link) and [part2](https://drive.google.com/file/d/1HqcPe5F9nX7cxBsWWo9IDkkAczVGRbBd/view?usp=sharing)).


The DocTamper dataset is only available for non-commercial use, you can request a password for it by sending an email to 202221012612@mail.scut.edu.cn explaining the purpose.


I delay the release of training codes as forced by my supervisor and the cooperative enterprise who bought them. My training pipline for DocTamper dataset and the IoU metric heavily brought from a famous project in this area, the results of  the paper can be easily re-produced with [it](https://github.com/DLLXW/data-science-competition/blob/main/tianchi/ImageForgeryLocationChallenge/utils/deeplearning_qyl.py), you just need to adjust the loss functions and the learing rate decay curve. I also used its [augmentation pipline](https://github.com/DLLXW/data-science-competition/blob/main/tianchi/ImageForgeryLocationChallenge/dataset/RSCDataset.py) except for (RandomBrightnessContrast, ShiftScaleRotate, CoarseDropout).


Open Source Scheme: <br>
1、Inference models and codes: June, 2023. <br>
2、Training codes: TBD. <br>
3、Data synthesis code: After we complete the expanded version of this work. <br>


Any question about this work please contact 202221012612@mail.scut.edu.cn.

