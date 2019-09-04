import cv2
from keras.datasets import mnist

(_, _), (test_images, test_labels) = mnist.load_data()


label_dict = {}

for i in range(20):
    print(test_labels[i])
    count = label_dict.get(test_labels[i], 0)
    if count > 0:
        count = count + 1
    else:
        count = 1

    label_dict[test_labels[i]] = count

    cv2.imwrite(str(test_labels[i]) + "_" + str(count) + ".png", test_images[i])

'''
Lấy ra một số ảnh để test thử
'''