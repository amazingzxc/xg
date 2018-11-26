
import os
import tensorflow as tf
from model.generator import encoder,decoder
from model.discriminator import discriminator
from model.data import Data

class XGAN(object):
    def __init__(self,options, encoder, decoder, discriminator, data):
        self.options=options
        self.encoder = encoder
        self.decoder=decoder
        self.discriminator = discriminator
        self.data = data

        # condition
        self.size = self.data.size
        self.channel = self.options.input_c_dim

        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])

        # nets
            #G_part(originDomain_temDomain)
        tmp_feature1_1,self.domain_X_encode = self.encoder(self.X,self.options,reuse=False, name="encoderX")
        tmp_feature2_1,self.domain_Y_encode = self.encoder(self.Y,self.options,reuse=False, name="encoderY")

        self.domain_X_decode_Y = self.decoder(tmp_feature1_1,self.domain_X_encode,self.options, reuse=False, name="decoderY")
        self.domain_Y_decode_X = self.decoder(tmp_feature2_1,self.domain_Y_encode,self.options, reuse=False, name="decoderX")

        _,self.domain_X_encode_1 = self.encoder(self.domain_X_decode_Y,self.options, reuse=True, name="encoderY")
        _,self.domain_Y_encode_1 = self.encoder(self.domain_Y_decode_X,self.options, reuse=True, name="encoderX")

            #D_part
        self.DY_fake = self.discriminator(self.domain_X_decode_Y,self.options, reuse=False, name="discriminatorYdomain")
        self.DX_fake = self.discriminator(self.domain_Y_decode_X,self.options, reuse=False, name="discriminatorXdomain")
        self.DX_real = self.discriminator(self.X,self.options, reuse=True, name="discriminatorXdomain")
        self.DY_real = self.discriminator(self.Y,self.options, reuse=True, name="discriminatorYdomain")
        # loss
            #G_part
        self.GX2Yloss = tf.reduce_mean((self.domain_X_encode - self.domain_X_encode_1) ** 2)
        self.GY2Xloss = tf.reduce_mean((self.domain_Y_encode - self.domain_Y_encode_1) ** 2)
        self.G_dx_yloss = tf.reduce_mean((self.DY_fake - tf.ones_like(self.DY_fake)) ** 2)
        self.G_dy_xloss = tf.reduce_mean((self.DX_fake - tf.ones_like(self.DX_fake)) ** 2)

        self.G_loss=self.GX2Yloss+self.GY2Xloss+self.G_dx_yloss+self.G_dy_xloss
            #D_part
        self.D_Xloss=tf.reduce_mean((self.DX_fake - tf.zeros_like(self.DX_fake)) ** 2)+\
                    tf.reduce_mean((self.DX_real - tf.ones_like(self.DX_real)) ** 2)

        self.D_Yloss = tf.reduce_mean((self.DY_fake - tf.zeros_like(self.DY_fake)) ** 2) + \
                       tf.reduce_mean((self.DY_real - tf.ones_like(self.DY_real)) ** 2)

        self.D_loss=self.D_Xloss+self.D_Yloss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'encoder' in var.name or 'decoder' in var.name]

        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss,
                                                                                       var_list=self.d_vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss,
                                                                                       var_list=self.g_vars)



    def train(self):
        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(1,self.options.epoch+1):
            # update D
            for _ in range(1):
                X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
                self.sess.run(
                    self.D_solver,
                    feed_dict={self.X: X_domain_image, self.Y: Y_domain_image})
            # update G
            for _ in range(1):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.X: X_domain_image, self.Y: Y_domain_image}
                )
            '''
                # fake img label to train G
                self.sess.run(
                    self.C_fake_solver,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
            '''
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                D_loss_curr = self.sess.run(
                    self.D_loss,
                    feed_dict={self.X: X_domain_image, self.Y: Y_domain_image})
                G_loss_curr = self.sess.run(
                    self.G_loss,
                    feed_dict={self.X: X_domain_image, self.Y: Y_domain_image})
                print('Iter: %d; D loss: %10.3f; G_loss: %10.3f'%(epoch,D_loss_curr,G_loss_curr))
                # if epoch % 500 == 0:
                #     y_s = sample_y(16, self.y_dim, fig_count % 10)
                #     samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})
                #
                #     fig = self.data.data2fig(samples)
                #     plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count % 10)),
                #                 bbox_inches='tight')
                #     fig_count += 1
                #     plt.close(fig)

            if (epoch % 2000 == 0 ) or epoch == (self.options.epoch):
                self.saver.save(self.sess, os.path.join(os.getcwd(),'model_save', 'xgan.ckpt'))
                print('model save at %s'%os.path.join(os.getcwd(),'model_save', 'xgan.ckpt'))

    def test(self,model_name):
        with tf.Graph().as_default():
            # self.build_graph()
            self.saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'model_save' , (model_name+'.ckpt.meta')))
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # config.allow_soft_placement = True
            # config.log_device_placement = False
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess,os.path.join(os.getcwd(), 'model_save' , (model_name+'.ckpt')))
            X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
            Y_domain_image_ = self.sess.run(self.domain_X_decode_Y,feed_dict={self.X: X_domain_image, self.Y: Y_domain_image})
            return Y_domain_image_

if __name__=='__main__':
    from collections import namedtuple

    data=Data()
    OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                                  gf_dim df_dim output_c_dim is_training')
    options = OPTIONS._make((2, 2,
                             2, 2, 2,
                              'train'))

    xgan=XGAN(options,encoder,decoder,discriminator,data)