import matplotlib.pyplot as plt
import tensorflow as tf


with tf.Session() as sess:
    image_raw = tf.gfile.FastGFile("0.jpg", "rb").read()
    image_raw = tf.image.decode_jpeg(image_raw)
    image_gray = sess.run(tf.image.rgb_to_grayscale(image_raw))
    image_raw = sess.run(image_raw)
    print(image_raw.shape)
    print(image_gray.shape)
    plt.imshow(image_gray[:,:,0], "gray")
    plt.show()