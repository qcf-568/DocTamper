# DocTamper

This is the official repository of the paper Towards Robust Tampered Text Detection in Document Image: New dataset and New Solution. [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.pdf).

The DocTamper dataset is now avaliable at [BaiduDrive](https://pan.baidu.com/s/1nEEnq1ZWIem7wnkQ1YdTNw?pwd=od9k) and [Kaggle](https://www.kaggle.com/datasets/dinmkeljiame/doctamper/data).


The DocTamper dataset is only available for non-commercial use, you can request a password for it by sending an email  __with education email__ to 202221012612@mail.scut.edu.cn explaining the purpose.

To visualize the images and their corresponding ground-truths from the provided .mdb files, you can run this command "python vizlmdb.py --input DocTamperV1-FCD --i 0".

---
The official implementation of the paper  Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution is in the "models" directory.

Open Source Scheme: <br>

1、Inference [models and code](https://github.com/qcf-568/DocTamper/tree/main/models)

2、Training code: contact 202221012612@mail.scut.edu.cn.

3、[Data synthesis code](https://github.com/qcf-568/DocTamper/tree/main/stg)

---

### The DocTamper dataset does not cover AIGC text tampering, but such a scenario is sufficiently covered by our [new work](https://github.com/qcf-568/OSTF).

---

Any question about this work please contact 202221012612@mail.scut.edu.cn.

---

If you find this work useful in your research, please consider citing:
```
@inproceedings{qu2023towards,
  title={Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution},
  author={Qu, Chenfan and Liu, Chongyu and Liu, Yuliang and Chen, Xinhong and Peng, Dezhi and Guo, Fengjun and Jin, Lianwen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5937--5946},
  year={2023}
}
```
