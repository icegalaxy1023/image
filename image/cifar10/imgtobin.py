# encoding:utf-8
"""
关于图片的一些操作：
①图片转化为数组并存为二进制文件；
②从二进制文件中读取数据并重新恢复为图片
"""

from __future__ import print_function
import numpy as np
import PIL.Image as Image
import pickle
import matplotlib.image as plimg
class Imageconversion(object):
    image_base_path = "C:/image/cifar10/picture/"
    data_base_path = "C:/image/cifar10/datasets/cifar-10-batches/"

    def image_to_array(self, filenames):
        """
        图片转化为数组并存为二进制文件；
        :param filenames:文件列表
        :return:
        """
        n = filenames.__len__()  # 获取图片的个数
        result = np.array([])  # 创建一个空的一维数组
        image = np.array([])

        print(u"开始将图片转为数组")
        for i in range(n):
            image = Image.open(self.image_base_path + filenames[i])
            image = image.resize((32, 32),Image.ANTIALIAS)            
            r, g, b = image.split()  # rgb通道分离
            # 注意：下面一定要reshpae(1024)使其变为一维数组，否则拼接的数据会出现错误，导致无法恢复图片
            #将PILLOW图像转成数组
            r_arr = plimg.pil_to_array(r)
            g_arr = plimg.pil_to_array(g)
            b_arr = plimg.pil_to_array(b)
            r_arr = np.array(r).reshape(1024)
            g_arr = np.array(g).reshape(1024)
            b_arr = np.array(b).reshape(1024)
            # 行拼接，类似于接火车；最终结果：共n行，一行3072列，为一张图片的rgb值
            image_arr = np.concatenate((r_arr, g_arr, b_arr))
            result = np.concatenate((result, image_arr))            

        print(u"转为数组成功，开始保存到文件")  
        # 构造字典,所有的图像诗句都在arr数组里,我这里是个以为数组,目前并没有存label
        contact = {'labels':10,'data':result}
        file_path = self.data_base_path + "data_batch_6"
        with open(file_path, mode='wb') as f:
#            p.dump(result, f)
             pickle.dump(contact, f)#把字典存到文本中去
        print(u"保存文件成功")



if __name__ == "__main__":
    imgbin = Imageconversion()
    images = []
    for j in range(2):
        images.append('img'+str(j) + ".jpg")
    imgbin.image_to_array(images)

