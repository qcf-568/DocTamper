#### The weights of the DTD for DocTamper is now avaliable at https://pan.baidu.com/s/166A9pentu3mwewY-79qHKg?pwd=vmhb  <Br/>

#### An example of dirs structure: <Br/>

---DocTamper <Br/>
&emsp;| <Br/>
&emsp;---qt_table.pk <Br/>
&emsp;---fph.py <Br/>
&emsp;---dtd.py <Br/>
&emsp;---swins.py <Br/>
&emsp;---eval_dtd.py <Br/>
&emsp;---pths <Br/>
&emsp;&emsp;| <Br/>
&emsp;&emsp;---dtd.pth <Br/>
&emsp;&emsp;---swin_imagenet.pt <Br/>
&emsp;&emsp;---vph_imagenet.pt <Br/>
&emsp;---DocTamperV1-FCD <Br/>
&emsp;&emsp;| <Br/>
&emsp;&emsp; ---data.mdb <Br/>
&emsp;&emsp; ---lock.mdb <Br/>
&emsp;---DocTamperV1-SCD <Br/>
&emsp;---DocTamperV1-TestingSet <Br/>
&emsp;---pks <Br/>
      
#### An example command to reproduce the results: <Br/>

CUDA_VISIBLE_DEVICES=0 python eval_dtd.py --lmdb_name DocTamperV1-FCD --pths/dtd.pth --minq 75 <Br/> <Br/> <Br/>


The weights and inference codes for T-SROIE dataset will be updated within a few days.
