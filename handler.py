try:
    import unzip_requirements
except ImportError:
    pass

import json
import wget
import boto3

from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.ssd import load_test
from gluoncv.model_zoo import get_model
from gluoncv.utils.viz import plot_bbox

# use mobilenet because it's small. lambda only has 512mb of space which isn't big enough
# to download ssd or frcnn models and unzip them. hosting them on s3 unzipped is probably
# a solution
ssdnet = get_model('ssd_512_mobilenet1.0_voc',
                   pretrained=True, root='/tmp/models')
s3 = boto3.client('s3')
score_threshold = 0.5


def getS3Url(s3_bucket_name, key_name):
    object_url = "https://s3-us-west-2.amazonaws.com/{0}/{1}".format(
        s3_bucket_name,
        key_name)
    return object_url


def detect(event, context):
    # get the url
    data = json.loads(event['body'])

    if 'url' not in data:
        response = {
            "statusCode": 500,
            "body": "Please specify a url"
        }
        return response

    url = data['url']
    # download the image
    urlSplit = url.split('/')
    fileName = urlSplit[-1]
    filePath = wget.download(url, out="/tmp/{0}".format(fileName))

    # classify the image
    x, img = load_test(filePath, short=512)
    classes, scores, bbox = ssdnet(x)

    results = []

    # for each result, we'll take the each
    # them if their score is greater than a given threshold
    for i in range(len(scores[0])):
        if float(scores[0][i].asnumpy().tolist()[0]) > score_threshold:
            results.append({
                "class": ssdnet.classes[int(classes[0][i].asnumpy().tolist()[0])],
                "score": float(scores[0][i].asnumpy().tolist()[0]),
                "bbox": bbox[0][i].asnumpy().tolist()
            })

    # plot the box of the image and then store it in S3
    plot_bbox(img, bbox[0], scores[0], classes[0], class_names=ssdnet.classes)

    tmpOutPath = "/tmp/detect_{0}".format(fileName)
    plt.savefig(tmpOutPath)

    s3_key = "images/detect_{0}".format(fileName)
    s3_bucket_name = "gudongfeng.me"
    s3.upload_file(tmpOutPath, s3_bucket_name, s3_key)

    body = {
        "bounding_boxes": results,
        "s3_url": getS3Url(s3_bucket_name, s3_key)
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': True,
            'Content-Type': 'application/json'
        },
    }

    return response
