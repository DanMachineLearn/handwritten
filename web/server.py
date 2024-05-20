from bottle import Bottle, request, run, static_file, response
import base64
from io import BytesIO
from PIL import Image
import os
import sys
sys.path.append('.')

from using.handwritten_using_googlenet import ProductGooglenet

app = Bottle()
models = [ProductGooglenet()]

# 路由：提供静态文件
@app.route('/')
def serve_index():
    return static_file('index.html', root='./web')

# 路由：获取模型列表
@app.route('/api/models')
def get_models():

    names = []
    for m in models:
        names.append(m.model_name)

    # 模拟模型列表
    response.content_type = 'application/json'
    return {"models": names}

# 路由：处理上传
@app.post('/api/upload')
def upload_image():
    try:
        data_url = request.forms.get('image')
        model = request.forms.get('model')
        # 去掉 data:image/jpeg;base64, 前缀
        base64_data = data_url.split(',')[1]
        # 解码 base64 数据
        img_data = base64.b64decode(base64_data)
        # 将字节数据转换为图像
        # with open('uploaded.jpg', 'wb') as f:
        #     f.write(img_data)
        image = Image.open(BytesIO(img_data))
        # image.show()
        # # 保存图像为 JPEG 文件
        image.save(f"images/t.png")

        ## TODO 使用模型处理图片数据
        model_net = None
        for m in models:
            if m.model_name == model:
                model_net = m
                break;
        labels = model_net.check("images/t.png")

        # 模拟处理并返回标签
        return {"status": "success", "labels": labels}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    run(app, host='localhost', port=8080)
