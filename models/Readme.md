The weights of the DTD for DocTamper is now avaliable at https://pan.baidu.com/s/15BrJQTTEuUzLs1X2JUUauw?pwd=fypf <Br/>

An example of dirs structure: <Br/>

---DocTamper
      | <Br/>
      ---qt_table.pk <Br/>
      ---fph.py <Br/>
      ---dtd.py <Br/>
      ---swins.py <Br/>
      ---eval_dtd.py <Br/>
      ---dtd.pth <Br/>
      ---DocTamperV1-FCD <Br/>
            |
            ---data.mdb <Br/>
            ---lock.mdb <Br/>
      ---DocTamperV1-SCD <Br/>
      ---DocTamperV1-TestingSet <Br/>
      
An example command to reproduce the results: <Br/>

CUDA_VISIBLE_DEVICES=0 python eval_dtd.py --lmdb_name DocTamperV1-FCD --pth dtd.pth --minq 75 <Br/>
