from trainer import model
import tensorflow as tf

CHANNEL = model.CHANNEL
HEIGHT = model.HEIGHT
WIDTH = model.WIDTH

if __name__ == '__main__':
    random_dim = 100
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    # # wgan
    # fake_image = generator(random_input, random_dim, is_train)
    # real_result = discriminator(real_image, is_train)
    # fake_result = discriminator(fake_image, is_train, reuse=True)
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # variables_to_restore = tf.contrib.slim.get_variables_to_restore(include=['gen'])
    # print(variables_to_restore)
    # saver = tf.train.Saver(variables_to_restore)
    # ckpt = tf.train.latest_checkpoint('./model/' + version)
    # saver.restore(sess, ckpt)
