import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


with tf.Session() as sess:
    image_raw = tf.gfile.FastGFile("0.jpg", "rb").read()
    image_raw = tf.image.decode_jpeg(image_raw)
    image_new = tf.image.resize_images(image_raw, [300,300], method=1)
    image_gray = sess.run(tf.image.rgb_to_grayscale(image_raw))
    image_raw = sess.run(image_raw)
    image_new = sess.run(image_new)
    print(image_raw.shape)
    print(image_gray.shape)
    # plt.imshow(image_new, "new")
    # plt.imshow(image_gray[:,:,0], "gray")
    # plt.show()
    cv2.imshow("gray", image_gray)
    cv2.imshow("new", image_new)
    cv2.waitKey()
