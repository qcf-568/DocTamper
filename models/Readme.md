The weights of the DTD for DocTamper is now avaliable at https://pan.baidu.com/s/15BrJQTTEuUzLs1X2JUUauw?pwd=fypf <Br/>

An example of dirs structure: <Br/>

---DocTamper
&emsp;| <Br/>
&emsp;---qt_table.pk <Br/>
&emsp;---fph.py <Br/>
&emsp;---dtd.py <Br/>
&emsp;---swins.py <Br/>
&emsp;---eval_dtd.py <Br/>
&emsp;---dtd.pth <Br/>
&emsp;---DocTamperV1-FCD <Br/>
&emsp;&emsp;|
&emsp;&emsp; ---data.mdb <Br/>
&emsp;&emsp; ---lock.mdb <Br/>
&emsp;---DocTamperV1-SCD <Br/>
&emsp;---DocTamperV1-TestingSet <Br/>
      
An example command to reproduce the results: <Br/>

CUDA_VISIBLE_DEVICES=0 python eval_dtd.py --lmdb_name DocTamperV1-FCD --pth dtd.pth --minq 75 <Br/>
