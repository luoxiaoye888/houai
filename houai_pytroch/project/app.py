from flask import Flask, jsonify, request
import sqlite3 as lite
import os
import cv2
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import backend as K
from datetime import datetime


app = Flask('__name__')  # 初始化App
app.config['JSON_AS_ASCII'] = False

db_path = os.path.join(os.getcwd(), 'apple.db')  # 数据库路径
model_path = os.path.join(os.getcwd(), 'models/densenet121.h5')  # 模型路径
app = Flask(__name__)

# 下面两行不是必需的，有的GPU需要指定显存使用方式，下面的2句只有在有异常的情况下使用
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config=tf.config.experimental.set_memory_growth(physical_devices[0],True)

# model = load_model('./models/densenet121.h5')  # 加载模型

# 图片大小
im_width = 256
im_height = 256
smooth = 100


# 定义Dice系数
def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return ((2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

# 定义损失函数
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# 定义iou函数
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

# 这个方法从全部的最优化节点中读取数据,因为有自定义的函数,所以加载时会有异常,
# 要使用custom_objects参数,通过字典的方式加载自定义的函数
model = tf.keras.models.load_model('houai90-300.h5',
                                   custom_objects={'dice_coef':dice_coef,
                                                   'dice_coef_loss': dice_coef_loss,
                                                   'iou':iou})
#model = model.load_weights('model/myUnet-300.h5')

# 从前台传来的值直接就是已经处理过的 ndarray 的数据
def fun_predict_houai(img1, imgname):
    #print(imgPath)
    prediction_overlap = []
    # img1 = cv2.imread(df_test['filename'].iloc[index])
    #img1 = cv2.imread("dataset/srcs/01_0064.jpg")
    #img1 = cv2.imread(img1)
    img1 = cv2.resize(img1, (im_height, im_width))
    #img = img1 / 255
    # 从前台传来值已经进行过处理了，不用再 /255 转换值了，否则会出现错误
    img = img1
    img = img[np.newaxis, :, :, :]
    prediction = model.predict(img)
    prediction = np.squeeze(prediction)
    #ground_truth = cv2.resize(cv2.imread(df_test['mask'].iloc[index]),(256,256)).astype("uint8")
    ground_truth = cv2.resize(cv2.imread("F:/lwx/project/hello2/labels_show/"+imgname),(256,256)).astype("uint8")
    ground_truth = cv2.cvtColor(ground_truth,cv2.COLOR_BGR2GRAY)
    _, thresh_gt = cv2.threshold(ground_truth, 127, 255, 0)
    # a,contours_gt,b = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 新版的opencv中的cv2.findContours只返回两个参数,一个是边框,一个层次关系
    contours_gt, a = cv2.findContours(thresh_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlap_img = cv2.drawContours(img1, contours_gt, 0, (0, 255, 0),1)   # 真实值为绿色
    prediction_overlap.append(overlap_img)

    prediction[np.nonzero(prediction < 0.3)] = 0.0
    prediction[np.nonzero(prediction >= 0.3)] = 255.
    prediction = prediction.astype("uint8")
    _, thresh_p = cv2.threshold(prediction, 127, 255, 0)
    # a,contours_p, b = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_p, b = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    overlap_img = cv2.drawContours(img1, contours_p, 0, (255,0,0),1)   # 预测边框用某种红色
    prediction_overlap.append(overlap_img)

    # print("预测的对象个数：%d" % (len(prediction_overlap)))
    # 64 X 64 的区域，分辨经为8，最终保存 图片为512 * 512 像素
    plt.figure(figsize=(64,64))
    for i in range(len(prediction_overlap)):
        #plt.subplot(4,4,i+1)
        plt.imshow(prediction_overlap[i])
    # 时间字符串, 第预测一次生成一个新的图片，否则它会先找到显示上一次处理的图片，不显示新处理的图片
    now_time = datetime.now()
    str_time = now_time.strftime("%Y%m%d%H%M%S")
    pic_name = "Predict"+str_time+".png"
    picPath = "static/pic/"+pic_name
    # plt.savefig("static/Predict.png", dpi=8)
    # dpi 像素密度值，与区域值相乘得到像素值，bbox_inches 去掉图片的白色边框
    plt.savefig(picPath, dpi=8, bbox_inches='tight')
    #plt.show()
    return pic_name


@app.route('/')  # 根目录服务端点
def index():
    message = {'db': 'apple.db', 'model': 'densenet121.h5'}
    return jsonify(message), 200

# 查询苹果数据集所有记录
@app.route('/get_all_details', methods=['get'])
def get_all_details():
    all_records = []
    conn = lite.connect(db_path)  # 打开数据库
    with conn:
        cur = conn.cursor()
        sql = f"select * from apples"
        cur.execute(sql)
        records = cur.fetchall()
        if records:  # 找到记录
            for record in records:
                item = dict(zip(['id', 'name', 'feature', 'regular', 'cure', 'img_url'], record))
                all_records.append(item)
            return jsonify(all_records), 200
        else:  # 没找到
            return jsonify(message='数据集为空！'), 404

# 图像预处理
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['post'])
def predict():
    message = request.get_json(force=True)
    image = message['image']
    decode_image = base64.b64decode(image)
    image = Image.open(io.BytesIO(decode_image))
    processed_image = preprocess_image(image, target_size=(512, 512))
    prediction = model.predict(processed_image)[0].tolist()  # 预测
    response = {
        'prediction': {
            'healthy': prediction[0],
            'multiple_diseases': prediction[1],
            'rust': prediction[2],
            'scab': prediction[3]
        }
    }
    return jsonify(response), 200

# 喉癌病灶区预测服务方法方法
@app.route('/predict_houai', methods=['post'])
def predict_houai():
    message = request.get_json(force=True)
    # 前端页面传来的图片
    image = message['image']
    # 前端页面传来图片名称，方便后端调用真实值ground_truth进行对比
    imgname = message['imgName']
    decode_image = base64.b64decode(image)
    image = Image.open(io.BytesIO(decode_image))
    # base64的文件类型为：<class 'PIL.JpegImagePlugin.JpegImageFile'>
    # print("Base64图片类型：", type(image))
    # 目标图片为256 X 256 大小的
    processed_image = preprocess_image(image, target_size=(512, 512))

    img1 = np.asarray(processed_image)[0]
    #print(img1.shape)   # 图像的尺寸
    result_img = fun_predict_houai(img1, imgname)
    response = {
        'result_img': result_img
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)