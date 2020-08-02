"""
DOCSTRING
"""
import datetime
#import google.cloud.storage
#import google.oauth2.service_account
#import googleapiclient.discovery
#import googleapiclient.errors
import logging
import math
import numpy
import os
import pprint
import scipy.misc
import tensorflow
import time

logging.basicConfig(level=logging.INFO)
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

class GCloud:
    """
    DOCSTRING
    """
    def __init__(self):
        self.bucket_name = 'pokegan-data'
        self.json_path = 'My Project 58923-14efe3d8f5f7.json'
        self.package_name = 'pokegan-0.0.0.tar.gz'
        self.project_id = 'ivory-hallway-204216'

    def create_bucket(storage_client):
        """
        DOCSTRING
        """
        logging.info('bucket:creating')
        if storage_client.lookup_bucket(self.bucket_name):
            logging.info('bucket:already exists')
        else:
            storage_client.create_bucket(self.bucket_name)
            logging.info('bucket:created')
        return storage_client.bucket(self.bucket_name)

    def create_job():
        """
        DOCSTRING
        """
        logging.info('job:creating')
        credentials = \
            google.oauth2.service_account.Credentials.from_service_account_file(self.json_path)
        ml = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)
        project_path = 'projects/{}'.format(self.project_id)
        training_input = {
            'jobDir': 'gs://{}/model'.format(self.bucket_name),
            'packageUris': ['gs://{}/{}'.format(self.bucket_name, self.package_name)],
            'pythonModule': 'trainer.task',
            'region': 'us-central1',
            'runtimeVersion': '1.6',
            'scaleTier': 'BASIC'}
        now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        job_specs = {'jobId': 'pokegan_' + now, 'trainingInput': training_input}
        request = ml.projects().jobs().create(parent=project_path, body=job_specs)
        try:
            request.execute()
            logging.info('job:created')
        except googleapiclient.errors.HttpError as err:
            logging.error('job:{}'.format(err._get_reason()))

    def delete_bucket():
        """
        DOCSTRING
        """
        logging.info('bucket:deleting')
        credentials = \
            google.oauth2.service_account.Credentials.from_service_account_file(self.json_path)
        storage_client = google.cloud.storage.Client(self.project_id, credentials)
        for bucket in storage_client.list_buckets():
            storage_client.delete(bucket, force=True)
        logging.info('bucket:deleted')

    def upload_data():
        """
        DOCSTRING
        """
        src_dir = 'data/rgb_images'
        credentials = \
            google.oauth2.service_account.Credentials.from_service_account_file(self.json_path)
        storage_client = google.cloud.storage.Client(self.project_id, credentials)
        bucket = create_bucket(self.bucket_name, storage_client)
        logging.info('training data:uploading')
        for filename in os.listdir(src_dir):
            logging.info('training data:uploading:{}'.format(filename))
            blob = google.cloud.storage.Blob('data/' + filename, bucket)
            blob.upload_from_filename(os.path.join(src_dir, filename))
        logging.info('training data:uploaded')

    def upload_package():
        """
        DOCSTRING
        """
        credentials = \
            google.oauth2.service_account.Credentials.from_service_account_file(self.json_path)
        storage_client = google.cloud.storage.Client(self.project_id, credentials)
        bucket = create_bucket(self.bucket_name, storage_client)
        logging.info('package:uploading')
        blob = google.cloud.storage.Blob(self.package_name, bucket)
        blob.upload_from_filename(os.path.join('dist', self.package_name))
        logging.info('package:uploaded')

class Model:
    """
    Generate new kinds of Pokémon using a Wasserstein generative adversarial network.
    """
    def __init__(self):
        self.batch_size = 64
        self.channel = 3
        self.height = 128
        self.width = 128

    def discriminator(self, input, is_train, reuse=False):
        """
        DOCSTRING
        """
        channel_2, channel_4, channel_8, channel_16 = 64, 128, 256, 512
        with tensorflow.variable_scope('dis') as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = tensorflow.layers.conv2d(
                input, channel_2, 
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv1')
            bn1 = tensorflow.contrib.layers.batch_norm(
                conv1,
                is_training=is_train,
                epsilon=1e-5,
                decay = 0.9,
                updates_collections=None,
                scope = 'bn1')
            act1 = lrelu(conv1, n='act1')
            conv2 = tensorflow.layers.conv2d(
                act1, channel_4,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv2')
            bn2 = tensorflow.contrib.layers.batch_norm(
                conv2,
                is_training=is_train,
                epsilon=1e-5,
                decay=0.9,
                updates_collections=None,
                scope='bn2')
            act2 = lrelu(bn2, n='act2')
            conv3 = tensorflow.layers.conv2d(
                act2, channel_8,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv3')
            bn3 = tensorflow.contrib.layers.batch_norm(
                conv3,
                is_training=is_train,
                epsilon=1e-5,
                decay=0.9,
                updates_collections=None,
                scope='bn3')
            act3 = lrelu(bn3, n='act3')
            conv4 = tensorflow.layers.conv2d(
                act3, channel_16,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv4')
            bn4 = tensorflow.contrib.layers.batch_norm(
                conv4,
                is_training=is_train,
                epsilon=1e-5,
                decay=0.9,
                updates_collections=None,
                scope='bn4')
            act4 = lrelu(bn4, n='act4')
            dim = int(numpy.prod(act4.get_shape()[1:]))
            fc1 = tensorflow.reshape(act4, shape=[-1, dim], name='fc1')
            w2 = tensorflow.get_variable(
                'w2', shape=[fc1.shape[-1], 1], dtype=tensorflow.float32,
                initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
            b2 = tensorflow.get_variable(
                'b2', shape=[1], dtype=tensorflow.float32,
                initializer=tensorflow.constant_initializer(0.0))
            logits = tensorflow.add(tensorflow.matmul(fc1, w2), b2, name='logits')
            return logits

    def generator(input, random_dim, is_train, reuse=False):
        """
        DOCSTRING
        """
        channel_4, channel_8, channel_16, channel_32, channel_64 = 512, 256, 128, 64, 32
        s4 = 4
        output_dim = self.channel
        with tensorflow.variable_scope('gen') as scope:
            if reuse:
                scope.reuse_variables()
            w1 = tensorflow.get_variable(
                'w1',
                shape=[random_dim, s4*s4*channel_4],
                dtype=tensorflow.float32,
                initializer=tensorflow.truncated_normal_initializer(stddev=0.02))
            b1 = tensorflow.get_variable(
                'b1',
                shape=[channel_4*s4*s4],
                dtype=tensorflow.float32,
                initializer=tensorflow.constant_initializer(0.0))
            flat_conv1 = tensorflow.add(
                tensorflow.matmul(input, w1), b1, name='flat_conv1')
            conv1 = tensorflow.reshape(
                flat_conv1, shape=[-1, s4, s4, channel_4], name='conv1')
            bn1 = tensorflow.contrib.layers.batch_norm(
                conv1,
                is_training=is_train,
                epsilon=1e-5,
                decay=0.9,
                updates_collections=None,
                scope='bn1')
            act1 = tensorflow.nn.relu(bn1, name='act1')
            conv2 = tensorflow.layers.conv2d_transpose(
                act1, channel_8,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv2')
            bn2 = tensorflow.contrib.layers.batch_norm(
                conv2,
                is_training=is_train,
                epsilon=1e-5,
                decay=0.9,
                updates_collections=None, scope='bn2')
            act2 = tensorflow.nn.relu(bn2, name='act2')
            conv3 = tensorflow.layers.conv2d_transpose(
                act2, channel_16,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv3')
            bn3 = tensorflow.contrib.layers.batch_norm(
                conv3,
                is_training=is_train,
                epsilon=1e-5,
                decay=0.9,
                updates_collections=None,
                scope='bn3')
            act3 = tensorflow.nn.relu(bn3, name='act3')
            conv4 = tensorflow.layers.conv2d_transpose(
                act3, channel_32,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv4')
            bn4 = tensorflow.contrib.layers.batch_norm(
                conv4,
                is_training=is_train,
                epsilon=1e-5,
                decay = 0.9,
                updates_collections=None,
                scope='bn4')
            act4 = tensorflow.nn.relu(bn4, name='act4')
            conv5 = tensorflow.layers.conv2d_transpose(
                act4, channel_64,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv5')
            bn5 = tensorflow.contrib.layers.batch_norm(
                conv5,
                is_training=is_train,
                epsilon=1e-5,
                decay=0.9,
                updates_collections=None,
                scope='bn5')
            act5 = tensorflow.nn.relu(bn5, name='act5')
            conv6 = tensorflow.layers.conv2d_transpose(
                act5,
                output_dim,
                kernel_size=[5, 5],
                strides=[2, 2],
                padding="SAME",
                kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.02),
                name='conv6')
            act6 = tensorflow.nn.tanh(conv6, name='act6')
            return act6

    def lrelu(self, x, n, leak=0.2):
        """
        DOCSTRING
        """
        return tensorflow.maximum(x, leak * x, name=n)

    def process_data(self):
        """
        DOCSTRING
        """
        images = list()
        for file_name in os.listdir('data'):
            images.append(os.path.join('data', file_name))
        logging.debug(images)
        images = tensorflow.convert_to_tensor(images, dtype=tensorflow.string)
        images_queue = tensorflow.train.slice_input_producer([images])
        content = tensorflow.read_file(images_queue[0])
        image = tensorflow.image.decode_jpeg(content, channels=self.channel)
        with tensorflow.Session() as sess_1:
            logging.debug(sess_1.run(image))
        image = tensorflow.image.random_flip_left_right(image)
        image = tensorflow.image.random_brightness(image, max_delta=0.1)
        image = tensorflow.image.random_contrast(image, lower=0.9, upper=1.1)
        logging.debug(image.get_shape())
        size = [self.height, self.width]
        image = tensorflow.image.resize_images(image, size)
        image.set_shape([self.height, self.width, self.channel])
        logging.debug(image.get_shape())
        image = tensorflow.cast(image, tensorflow.float32)
        image = image / 255.0
        images_batch = tensorflow.train.shuffle_batch(
            [image], self.batch_size,
            capacity=200 + 3 * self.batch_size,
            min_after_dequeue=200, num_threads=4)
        num_images = len(images)
        return images_batch, num_images

    def task(self):
        """
        Trains a W-GAN to create new Pokemon.
        """
        random_dim = 100
        with tensorflow.variable_scope('input'):
            real_image = tensorflow.placeholder(
                tensorflow.float32,
                [None, self.height, self.width, self.channel],
                'real_image')
            random_input = tensorflow.placeholder(
                tensorflow.float32,
                [None, self.random_dim],
                'rand_input')
            is_train = tensorflow.placeholder(tensorflow.bool, name='is_train')
        # w-gan
        fake_image = model.generator(random_input, self.random_dim, is_train)
        real_result = model.discriminator(real_image, is_train)
        fake_result = model.discriminator(fake_image, is_train, reuse=True)
        d_loss = tensorflow.reduce_mean(fake_result) - tensorflow.reduce_mean(real_result)
        g_loss = -tensorflow.reduce_mean(fake_result)
        t_vars = tensorflow.trainable_variables()
        d_vars = [var for var in t_vars if 'dis' in var.name]
        g_vars = [var for var in t_vars if 'gen' in var.name]
        logging.debug(d_vars)
        learning_rate = 2e-4
        trainer_d = tensorflow.train.RMSPropOptimizer(
            learning_rate).minimize(d_loss, var_list=d_vars)
        trainer_g = tensorflow.train.RMSPropOptimizer(
            learning_rate).minimize(g_loss, var_list=g_vars)
        # clip discriminator weights
        d_clip = [v.assign(tensorflow.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
        batch_size = self.batch_size
        image_batch, samples_num = model.process_data()
        batch_num = int(samples_num / batch_size)

    def test(self):
        """
        DOCSTRING
        """
        random_dim = 100
        with tensorflow.variable_scope('input'):
            real_image = tensorflow.placeholder(
                tensorflow.float32,
                shape=[None, self.height, self.width, self.channel],
                name='real_image')
            random_input = tensorflow.placeholder(
                tensorflow.float32,
                shape=[None, random_dim],
                name='rand_input')
            is_train = tensorflow.placeholder(tensorflow.bool, name='is_train')
        # w-gan
        fake_image = generator(random_input, random_dim, is_train)
        real_result = discriminator(real_image, is_train)
        fake_result = discriminator(fake_image, is_train, reuse=True)
        sess = tensorflow.InteractiveSession()
        sess.run(tensorflow.global_variables_initializer())
        variables_to_restore = tensorflow.contrib.slim.get_variables_to_restore(include=['gen'])
        print(variables_to_restore)
        saver = tensorflow.train.Saver(variables_to_restore)
        ckpt = tensorflow.train.latest_checkpoint('model/' + version)
        saver.restore(sess, ckpt)

class Utils:
    """
    DOCSTRING
    """
    def __init__(self):
        pretty_printer = pprint.PrettyPrinter()
        get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])

    def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
        """
        DOCSTRING
        """
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.0))
        i = int(round((w - crop_w) / 2.0))
        return scipy.misc.imresize(
            x[j: j+crop_h, i: i+crop_w], [resize_h, resize_w])

    def get_image(
        image_path,
        input_height,
        input_width,
        resize_height=64,
        resize_width=64,
        crop=True,
        grayscale=False):
        """
        DOCSTRING
        """
        image = imread(image_path, grayscale)
        return transform(
            image, input_height, input_width, resize_height, resize_width, crop)

    def imread(path, grayscale=False):
        """
        DOCSTRING
        """
        if (grayscale):
            return scipy.misc.imread(path, flatten=True).astype(numpy.float)
        else:
            return scipy.misc.imread(path).astype(numpy.float)

    def imsave(images, size, path):
        """
        DOCSTRING
        """
        image = numpy.squeeze(merge(images, size))
        return scipy.misc.imsave(path, image)

    def inverse_transform(images):
        """
        DOCSTRING
        """
        return (images + 1.0) / 2.0

    def make_frame(self, t):
        """
        DOCSTRING
        """
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
            if true_image:
                return x.astype(numpy.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(numpy.uint8)

    def make_gif(images, fname, duration=2, true_image=False):
        """
        DOCSTRING
        """
        import moviepy.editor
        clip = moviepy.editor.VideoClip(self.make_frame, duration=duration)
        clip.write_gif(fname, fps=len(images) / duration)

    def merge(images, size):
        """
        DOCSTRING
        """
        h, w = images.shape[1], images.shape[2]
        if (images.shape[3] in (3, 4)):
            c = images.shape[3]
            img = numpy.zeros((h * size[0], w * size[1], c))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h: j * h + h, i * w: i * w + w, :] = image
            return img
        elif images.shape[3]==1:
            img = numpy.zeros((h * size[0], w * size[1]))
            for idx, image in enumerate(images):
                i = idx % size[1]
                j = idx // size[1]
                img[j * h: j * h + h, i * w: i * w + w] = image[:,:,0]
            return img
        else:
            raise ValueError('Images parameter must have HxW, HxWx3, or HxWx4')

    def merge_images(images, size):
        """
        DOCSTRING
        """
        return inverse_transform(images)
        
    def rename_images(self, src_dir, new_prefix):
        """
        DOCSTRING
        """
        for file_name in os.listdir(src_dir):
            os.rename(
                os.path.join(src_dir, file_name),
                os.path.join(src_dir, new_prefix + file_name[5:]))
            print(file_name + ' -> ' + new_prefix + file_name)

    def resize_images(self):
        """
        DOCSTRING
        """
        logging.info('images:resizing')
        src_path = 'data/original_images'
        dst_path = 'data/resized_images'
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for filename in os.listdir(src_path):
            img = PIL.Image.open(os.path.join(src_path, filename))
            img = img.resize((256, 256), PIL.Image.ANTIALIAS)
            img.save(os.path.join(dst_path, filename))
            logging.info('images:resizing:{}'.format(filename))
        logging.info('images:resizing completed')

    def rgba_to_rgb(self):
        """
        DOCSTRING
        """
        logging.info('images:converting to rgb')
        src_path = 'data/resized_images'
        dst_path = 'data/rgb_images'
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for filename in os.listdir(src_path):
            img = PIL.Image.open(os.path.join(src_path, filename))
            if img.mode == 'RGBA':
                logging.info('images:converting to rgb:{}'.format(filename))
                img.load() # required for png.split()
                split_img = PIL.Image.new('RGB', img.size, (0, 0, 0))
                split_img.paste(img, mask=img.split()[3]) # 3 is the alpha channel
                split_img.save(os.path.join(dst_path, filename.split('.')[0] + '.jpg'), 'JPEG')
            else:
                logging.info('images:converting to jpg:{}'.format(filename))
                img = img.convert('RGB')
                img.save(os.path.join(dst_path, filename.split('.')[0] + '.jpg'), 'JPEG')
        logging.info('images:conversion completed')

    def save_images(images, size, image_path):
        """
        DOCSTRING
        """
        return imsave(inverse_transform(images), size, image_path)

    def show_all_variables():
        """
        DOCSTRING
        """
        model_vars = tensorflow.trainable_variables()
        tensorflow.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def to_json(output_path, *layers):
        """
        DOCSTRING
        """
        with open(output_path, "w") as layer_f:
            lines = ""
            for w, b, bn in layers:
                layer_idx = w.name.split('/')[0].split('h')[1]
                B = b.eval()
                if "lin/" in w.name:
                    W = w.eval()
                    depth = W.shape[1]
                else:
                    W = numpy.rollaxis(w.eval(), 2, 0)
                    depth = W.shape[0]
                biases = {
                    "sy": 1, "sx": 1, "depth": depth,
                    "w": ['%.2f' % elem for elem in list(B)]}
                if bn != None:
                    gamma = bn.gamma.eval()
                    beta = bn.beta.eval()
                    gamma = {
                        "sy": 1, "sx": 1, "depth": depth,
                        "w": ['%.2f' % elem for elem in list(gamma)]}
                    beta = {
                        "sy": 1, "sx": 1, "depth": depth,
                        "w": ['%.2f' % elem for elem in list(beta)]}
                else:
                    gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                    beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                if "lin/" in w.name:
                    fs = list()
                    for w in W.T:
                        fs.append(
                            {"sy": 1,
                             "sx": 1,
                             "depth": W.shape[0],
                             "w": ['%.2f' % elem for elem in list(w)]})
                    lines += \
                        """
                        var layer_%s = {
                            "layer_type": "fc",
                            "sy": 1, "sx": 1,
                            "out_sx": 1, "out_sy": 1,
                            "stride": 1, "pad": 0,
                            "out_depth": %s, "in_depth": %s,
                            "biases": %s,
                            "gamma": %s,
                            "beta": %s,
                            "filters": %s}
                        """ % (
                            layer_idx.split('_')[0],
                            W.shape[1],
                            W.shape[0],
                            biases,
                            gamma,
                            beta,
                            fs)
                else:
                    fs = list()
                    for w_ in W:
                        fs.append(
                            {"sy": 5,
                             "sx": 5,
                             "depth": W.shape[3],
                             "w": ['%.2f' % elem for elem in list(w_.flatten())]})
                    lines += \
                        """
                        var layer_%s = {
                            "layer_type": "deconv",
                            "sy": 5, "sx": 5,
                            "out_sx": %s, "out_sy": %s,
                            "stride": 2, "pad": 1,
                            "out_depth": %s, "in_depth": %s,
                            "biases": %s,
                            "gamma": %s,
                            "beta": %s,
                            "filters": %s}
                        """ % (
                            layer_idx,
                            2**(int(layer_idx)+2),
                            2**(int(layer_idx)+2),
                            W.shape[0],
                            W.shape[3],
                            biases,
                            gamma,
                            beta,
                            fs)
            layer_f.write(" ".join(lines.replace("'","").split()))

    def transform(
        image,
        input_height,
        input_width,
        resize_height=64,
        resize_width=64,
        crop=True):
        """
        DOCSTRING
        """
        if crop:
            cropped_image = center_crop(
                image, input_height, input_width, resize_height, resize_width)
        else:
            cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
        return numpy.array(cropped_image) / 127.5 - 1.0

    def visualize(sess, dcgan, config, option):
        """
        DOCSTRING
        """
        image_frame_dim = int(math.ceil(config.batch_size**0.5))
        if option == 0:
            z_sample = numpy.random.uniform(
                -0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(
                samples, [image_frame_dim, image_frame_dim],
                'samples/test_%s.png' % time.strftime("%Y%m%d%H%M%S", time.gmtime()))
        elif option == 1:
            values = numpy.arange(0, 1, 1.0 / config.batch_size)
            for idx in range(100):
                print(" [*] %d" % idx)
                z_sample = numpy.zeros([config.batch_size, dcgan.z_dim])
                for kdx, z in enumerate(z_sample):
                    z[idx] = values[kdx]
                if config.dataset == "mnist":
                    y = numpy.random.choice(10, config.batch_size)
                    y_one_hot = numpy.zeros((config.batch_size, 10))
                    y_one_hot[numpy.arange(config.batch_size), y] = 1
                    samples = sess.run(
                        dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
                else:
                    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
                save_images(
                    samples,
                    [image_frame_dim, image_frame_dim],
                    'samples/test_arange_%s.png' % (idx))
        elif option == 2:
            values = numpy.arange(0, 1, 1.0 / config.batch_size)
            for idx in [numpy.random.randint(0, 99) for _ in range(100)]:
                print(" [*] %d" % idx)
                z = numpy.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
                z_sample = numpy.tile(z, (config.batch_size, 1))
                #z_sample = numpy.zeros([config.batch_size, dcgan.z_dim])
                for kdx, z in enumerate(z_sample):
                    z[idx] = values[kdx]
                if config.dataset == "mnist":
                    y = numpy.random.choice(10, config.batch_size)
                    y_one_hot = numpy.zeros((config.batch_size, 10))
                    y_one_hot[numpy.arange(config.batch_size), y] = 1
                    samples = sess.run(
                        dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
                else:
                    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
                try:
                    make_gif(samples, 'samples/test_gif_%s.gif' % (idx))
                except:
                    save_images(
                        samples,
                        [image_frame_dim, image_frame_dim],
                        'samples/test_%s.png' % time.strftime("%Y%m%d%H%M%S", time.gmtime()))
        elif option == 3:
            values = numpy.arange(0, 1, 1.0 / config.batch_size)
            for idx in range(100):
                print(" [*] %d" % idx)
                z_sample = numpy.zeros([config.batch_size, dcgan.z_dim])
                for kdx, z in enumerate(z_sample):
                    z[idx] = values[kdx]
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
                make_gif(samples, './samples/test_gif_%s.gif' % (idx))
        elif option == 4:
            image_set = list()
            values = numpy.arange(0, 1, 1.0 / config.batch_size)
            for idx in range(100):
                print(" [*] %d" % idx)
                z_sample = numpy.zeros([config.batch_size, dcgan.z_dim])
                for kdx, z in enumerate(z_sample):
                    z[idx] = values[kdx]
                image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
                make_gif(image_set[-1], 'samples/test_gif_%s.gif' % (idx))
            new_image_set = [merge(
                numpy.array([images[idx] for images in image_set]),
                [10, 10]) for idx in range(64) + range(63, -1, -1)]
            make_gif(new_image_set, 'samples/test_gif_merged.gif', duration=8)

if __name__ == '__main__':
    utils = Utils()
    utils.rename_images('images/original', 'original-')
