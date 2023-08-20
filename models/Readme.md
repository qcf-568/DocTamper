#### The weights of the DTD for DocTamper and T-SROIE are now avaliable at [Baidu Drive](https://pan.baidu.com/s/166A9pentu3mwewY-79qHKg?pwd=vmhb) or [Google Drive](https://drive.google.com/drive/folders/11Ep8PJIrlIveudQaRulDOBENHGqw762a?usp=sharing)  <Br/>

#### An Example [Colab Note Book](https://colab.research.google.com/drive/1rWaSKy2Rsy5welyvj6FbzF01o2zv8ips?usp=sharing) <Br/>

#### An example of dirs structure: <Br/>

---DocTamper/ <Br/>
&emsp;| <Br/>
&emsp;---qt_table.pk <Br/>
&emsp;---fph.py <Br/>
&emsp;---dtd.py <Br/>
&emsp;---swins.py <Br/>
&emsp;---eval_dtd.py <Br/>
&emsp;---pths/ <Br/>
&emsp;&emsp;| <Br/>
&emsp;&emsp;---dtd.pth <Br/>
&emsp;&emsp;---swin_imagenet.pt <Br/>
&emsp;&emsp;---vph_imagenet.pt <Br/>
&emsp;---DocTamperV1-FCD/ <Br/>
&emsp;&emsp;| <Br/>
&emsp;&emsp; ---data.mdb <Br/>
&emsp;&emsp; ---lock.mdb <Br/>
&emsp;---DocTamperV1-SCD/ <Br/>
&emsp;---DocTamperV1-TestingSet/ <Br/>
&emsp;---pks/ <Br/>
      
#### An example command to reproduce the results: <Br/>

<code>!CUDA_VISIBLE_DEVICES=0 python eval_dtd.py --lmdb_name DocTamperV1-FCD --pth pths/dtd_doctamper.pth --minq 75</code> <Br/> <Br/> <Br/>
