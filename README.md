# npCWAVE
### Numpy-only CWAVE implementation

- Usage:
  - Expected inputs: 
      todSAR(decimal), 
      lonSAR(decimal), 
      latSAR(decimal), 
      incidenceAngle(decimal), 
      sigma0(decimal), 
      normVar(decimal), 
      s0-s19 (all decimals and all are separate inputs), 
      sentinelType(binary where Sentinel A is represented as 1 and SB represented as 0)
  - Call as standalone script:
```
$ python DeepCWAVE.py 65713.0144444434 -103.07798767089844 4.205230236053467 24.22 -6.682582855224609 1.3222424983978271  11.708677291870117 0.9372023940086365 -0.42126742005348206 2.7530977725982666 -10.459186553955078 -0.8531363606452942 -2.4463205337524414 0.6690654158592224 -1.2713881731033325 2.9093339443206787 -3.833677291870117 0.07296248525381088 1.2206870317459106 -2.244554042816162 3.8712360858917236 -3.899672269821167 -3.1992101669311523 0.6109594106674194 -1.471483588218689 6.273694038391113 1.0
```
  - Import as python module:
```python
import DeepCWAVE as dc
y = dc.predict(x)
```
  
- Dependencies:
  - Numpy>=1.16.1
