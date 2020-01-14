"""
Task definition for Google ML Engine job.
The task trains a W-GAN to use in the creation of novel pokemon.

"""
import logging
import model
import numpy as np
import os
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

if __name__ == '__main__':
    BATCH_SIZE = model.BATCH_SIZE
    BUCKET_NAME = model.BUCKET_NAME
    CHANNEL = model.CHANNEL
    EPOCH = 5000
    HEIGHT = model.HEIGHT
    RANDOM_DIM = 100
    VERSION = 'new_pokemon'
    WIDTH = model.WIDTH

    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNEL], 'real_image')
        random_input = tf.placeholder(tf.float32, [None, RANDOM_DIM], 'rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    # wasserstein generative adversarial network
    fake_image = model.generator(random_input, RANDOM_DIM, is_train)
    real_result = model.discriminator(real_image, is_train)
    fake_result = model.discriminator(fake_image, is_train, reuse=True)
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)    # This optimizes the discriminator
    g_loss = -tf.reduce_mean(fake_result)    # This optimizes the generator
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    logging.debug(d_vars)
    learning_rate = 2e-4
    trainer_d = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
    batch_size = BATCH_SIZE
    image_batch, samples_num = model.process_data()
    batch_num = int(samples_num / batch_size)
#     # total_batch = 0
#     sess = tf.Session()
#     saver = tf.train.Saver()
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     # continue training
#     save_path = saver.save(sess, "/tmp/model.ckpt")
#     # ckpt = tf.train.latest_checkpoint('./model/' + version)
#     saver.restore(sess, save_path)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     print('total training sample num:%d' % samples_num)
#     print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
#     print('start training...')
#     for i in range(EPOCH):
#         print(i)
#         for j in range(batch_num):
#             print(j)
#             d_iters = 5
#             g_iters = 1
#             train_noise = np_random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np_float32)
#             for k in range(d_iters):
#                 print(k)
#                 train_image = sess.run(image_batch)
#                 # wgan clip weights
#                 sess.run(d_clip)
#                 # Update the discriminator
#                 _, dLoss = sess.run([trainer_d, d_loss], feed_dict={random_input: train_noise,
#                                                                     real_image: train_image,
#                                                                     is_train: True})
#             # Update the generator
#             for k in range(g_iters):
#                 # train_noise = np_random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np_float32)
#                 _, gLoss = sess.run([trainer_g, g_loss], feed_dict={random_input: train_noise,
#                                                                     is_train: True})
#             # print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)
#         # save check point every 500 epoch
#         if i % 500 == 0:
#             if not path.exists('./model/' + version):
#                 makedirs('./model/' + version)
#             saver.save(sess, './model/' +version + '/' + str(i))
#         if i % 50 == 0:
#             # save images
#             if not path.exists(newPoke_path):
#                 makedirs(newPoke_path)
#             sample_noise = np_random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np_float32)
#             imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
#             # imgtest = imgtest * 255.0
#             # imgtest.astype(np_uint8)
#             utils.save_images(imgtest, [8,8] ,newPoke_path + '/epoch' + str(i) + '.jpg')
#             print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
#     coord.request_stop()
#     coord.join(threads)
