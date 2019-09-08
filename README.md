# serverless-mxnet

MXNet 1.4.0 running on lambda with Gluon 0.3.0

The project is setup to use ssd_512_mobilenet1.0_voc for object detection

## Deploy

```
npm install
serverless deploy
```

## Invoke

```serverless invoke -f detect -p sample_data.json```

