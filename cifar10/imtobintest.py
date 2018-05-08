# encoding:utf-8

"""
关于图片的一些操作：
①图片转化为数组并存为二进制文件；
②从二进制文件中读取数据并重新恢复为图片
"""

from __future__ import print_function
from imgtobin import Imageconversion

my_operator = Imageconversion()
images = []
for j in range(60):
    images.append('img'+str(j) + ".jpg")
my_operator.image_to_array(images)

