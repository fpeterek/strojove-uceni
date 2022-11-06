```sh
cabal v2-run decisiontrees -- data/iris.csv 3 n int ";"
```

```
❯ cabal v2-run decisiontrees -- data/iris.csv 4 n int ";"
Up to date
[0,1,2,3]
4
{"attr":2,"th":2.24,"l":{"class":"0"},"r":{"attr":3,"th":1.5600001,"l":{"class":"1"},"r":{"attr":0,"th":5.9,"l":{"class":"1"},"r":{"class":"2"}}}}
1: (0.4814815,0.30081302)
2: (0.7777778,0.6422764)
3: (0.9259259,0.9512195)
4: (0.962963,0.8943089)
```
