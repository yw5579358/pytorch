import io
import json
import flask
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms, models, datasets
from torch.autograd import Variable

app = flask.Flask(__name__)
model = None
use_gpu = True


def load_model():
    global model
    model = models.resnet18()
    numb_firs = model.fc.in_features
    # 修改全连接输出，自己的分类个数
    model.fc = nn.Linear(numb_firs, 102)
    # 加载模型
    checkpoint = torch.load('../study/best.pt')
    model.load_state_dict(checkpoint['state_dict'])
    # 预测模式
    model.eval()

    if use_gpu:
        model.cuda()


# 数据预处理
def prepare_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)

    # 标准化
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.增加一个维度，用于按batch测试   本次这里一次测试一张
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return Variable(image, volatile=True)  # 不需要求导


@app.route('/predict', methods=['post'])
def predict():
    data = {"success": False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target_size=(64, 64))

            preds = F.softmax(model(image), dim=1)
            results = torch.topk(preds.cpu().data, k=3, dim=1)
            results = (results[0].cpu().numpy(), results[1].cpu().numpy())

            data['predictions'] = list()
            for prob, label in zip(results[0][0], results[1][0]):
                r = {"label": str(label), "probability": float(prob)}
                data['predictions'].append(r)

            data['success']=True
            return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    #先加载模型
    load_model()
    #再开启服务
    app.run(port=5012)
