
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tensorflow.contrib.layers as tcl
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

        self.build_xgan()

    def build_xgan(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])

        # nets
        # G_part(originDomain_temDomain)
        tmp_feature1_1, self.domain_X_encode = self.encoder(self.X, self.options, reuse=False, name="encoderX")
        tmp_feature2_1, self.domain_Y_encode = self.encoder(self.Y, self.options, reuse=False, name="encoderY")

        self.domain_X_decode_Y = self.decoder(tmp_feature1_1, self.domain_X_encode, self.options, reuse=False,
                                              name="decoderY")
        self.domain_Y_decode_X = self.decoder(tmp_feature2_1, self.domain_Y_encode, self.options, reuse=False,
                                              name="decoderX")

        tmp_feature1_2, self.domain_X_encode_1 = self.encoder(self.domain_X_decode_Y, self.options, reuse=True, name="encoderY")
        tmp_feature2_2, self.domain_Y_encode_1 = self.encoder(self.domain_Y_decode_X, self.options, reuse=True, name="encoderX")

        self.domain_X_decode_X_1 = self.decoder(tmp_feature1_2, self.domain_X_encode_1, self.options, reuse=True,
                                              name="decoderX")
        self.domain_Y_decode_Y_1 = self.decoder(tmp_feature2_2, self.domain_Y_encode_1, self.options, reuse=True,
                                              name="decoderY")
        # D_part
        self.DY_fake = self.discriminator(self.domain_X_decode_Y, self.options, reuse=False,
                                          name="discriminatorYdomain")
        self.DX_fake = self.discriminator(self.domain_Y_decode_X, self.options, reuse=False,
                                          name="discriminatorXdomain")
        self.DX_real = self.discriminator(self.X, self.options, reuse=True, name="discriminatorXdomain")
        self.DY_real = self.discriminator(self.Y, self.options, reuse=True, name="discriminatorYdomain")
        #真实图片x回来的结果
        self.DX_real_x = self.discriminator(self.domain_X_decode_X_1, self.options, reuse=True, name="discriminatorXdomain")
        self.DY_real_x = self.discriminator(self.domain_Y_decode_Y_1, self.options, reuse=True, name="discriminatorYdomain")
        # C part
        self.classifier_score_X_real=self.classifier(self.domain_X_encode,reuse=False)
        self.classifier_score_Y_real = self.classifier(self.domain_Y_encode,reuse=True)
        self.classifier_score_X_fake = self.classifier(self.domain_Y_encode_1,reuse=True)
        self.classifier_score_Y_fake = self.classifier(self.domain_X_encode_1,reuse=True)

        # loss
        # G_part
        #Lrec
        self.GXrealloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.domain_X_encode_1, labels=self.domain_X_encode))
        self.GYrealloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.domain_Y_encode_1, labels=self.domain_Y_encode))

        self.GYfakeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_fake, labels=tf.ones_like(self.DY_fake)))
        self.GXfakeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_fake, labels=tf.ones_like(self.DX_fake)))

        self.GXrealxloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_real_x, labels=tf.ones_like(self.DX_real_x)))
        self.GYrealxloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_real_x, labels=tf.ones_like(self.DY_real_x)))

        #G_C_loss
        self.G_CXloss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_X_fake,
                                                                           labels=tf.ones_like(self.classifier_score_X_fake)))
        self.G_CYloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_Y_fake,
                                                                             labels=tf.ones_like(self.classifier_score_Y_fake)))
        self.G_loss = self.GXrealloss + self.GYrealloss + \
                      0.5*self.GYfakeloss + 0.5*self.GXfakeloss + \
                      0.5 *self.GXrealxloss + 0.5*self.GYrealxloss+\
                      0.5*self.G_CXloss + 0.5*self.G_CYloss
        # D_part
        self.D_Xloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_fake, labels=tf.zeros_like(self.DX_fake)))+\
                       tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_real, labels=tf.ones_like(self.DX_real)))

        self.D_Yloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_fake, labels=tf.zeros_like(self.DY_fake))) + \
                       tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_real, labels=tf.ones_like(self.DY_real)))
        #D_C_loss
        self.D_CXloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_X_real,
                                                                                    labels=tf.ones_like(self.classifier_score_X_real)))
        self.D_CXloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_X_fake,
                                                                                    labels=tf.zeros_like(self.classifier_score_X_fake)))
        self.D_CYloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_Y_real,
                                                                                    labels=tf.ones_like(self.classifier_score_Y_real)))
        self.D_CYloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_Y_fake,
                                                                                    labels=tf.zeros_like(self.classifier_score_Y_fake)))

        self.D_loss = self.D_Xloss + self.D_Yloss +\
            0.5*self.D_CXloss_real + 0.5*self.D_CXloss_fake +\
            0.5*self.D_CYloss_real + 0.5*self.D_CYloss_fake

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'encoder' in var.name or 'decoder' in var.name]

        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss,
                                                                                       var_list=self.d_vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss,
                                                                                       var_list=self.g_vars)
    def build_Xgan_gpu(self):
        pass

    def classifier(self,input,reuse=False, name="discriminator_classifier"):
        with tf.variable_scope(name):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            hidden_1 = tcl.fully_connected(input, 128, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            hidden_2 = tcl.fully_connected(hidden_1, 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            output = tcl.fully_connected(hidden_2, 1, activation_fn=None)  # bin classes
            return output

    def train(self):
        last_G_loss=0.0
        last_D_loss=0.0
        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())
        X_domain_image_val,Y_domain_image_val=self.data.validation_data()

        for epoch in range(1,self.options.epoch+1):
            # update D
            for _ in tqdm(range(int(self.data._num_examples/self.options.batch_size))):
                X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
                self.sess.run(
                    self.D_solver,
                    feed_dict={self.X: X_domain_image, self.Y: Y_domain_image})
            # update G
            # for _ in range(int(self.data._num_examples/self.options.batch_size)):
            #     X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
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
            if epoch % 5 == 0 or epoch < 100:
                D_loss_curr,G_loss_curr=0.0,0.0
                for i in tqdm(range(X_domain_image_val.shape[0])):

                    D_loss_curr += self.sess.run(
                        self.D_loss,
                        feed_dict={self.X: X_domain_image_val[i][np.newaxis,:,:,:], self.Y: Y_domain_image_val[i][np.newaxis,:,:,:]})
                    G_loss_curr += self.sess.run(
                        self.G_loss,
                        feed_dict={self.X: X_domain_image_val[i][np.newaxis,:,:,:], self.Y: Y_domain_image_val[i][np.newaxis,:,:,:]})
                print('Iter: %d; D loss: %10.3f; G_loss: %10.3f'%(epoch,
                                                                  D_loss_curr/X_domain_image_val.shape[0],
                                                                  G_loss_curr/X_domain_image_val.shape[0]))
                # if epoch % 500 == 0:
                #     y_s = sample_y(16, self.y_dim, fig_count % 10)
                #     samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})
                #
                #     fig = self.data.data2fig(samples)
                #     plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count % 10)),
                #                 bbox_inches='tight')
                #     fig_count += 1
                #     plt.close(fig)

            #if  epoch%(self.options.save_freq) == 0:
            if epoch in [10,20,50,100,150,200]:
                self.saver.save(self.sess, os.path.join(os.getcwd(),'model_save', 'xgan%s_d%s_g%s.ckpt'%(epoch,D_loss_curr,G_loss_curr)))
                print('model save at %s'%os.path.join(os.getcwd(),'model_save', 'xgan.ckpt'))

    def test(self,model_name):
        # with tf.Graph().as_default():#加了就出问题。。。。？？？？？！！！！！！
        self.saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'model_save' , (model_name+'.ckpt.meta')))
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess,os.path.join(os.getcwd(), 'model_save' , (model_name+'.ckpt')))
        X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
        Y_domain_image_,X_domain_image_ = self.sess.run([self.domain_X_decode_Y,self.domain_X_decode_X_1],
                                        feed_dict={self.X: X_domain_image,
                                                   self.Y: Y_domain_image}
                                        )
        return Y_domain_image_,X_domain_image_
        # 连同图结构一同加载
        # ckpt = tf.train.get_checkpoint_state('./model_save/')
        # saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        #     X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
        #     Y_domain_image_ = sess.run(self.domain_X_decode_Y,
        #                                feed_dict={'Placeholder:0': X_domain_image,
        #                                           'Placeholder_1:0': Y_domain_image}
        #                                )
        # return Y_domain_image_

if __name__=='__main__':
    from collections import namedtuple

    data=Data()
    OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                                  gf_dim df_dim output_c_dim is_training')
    options = OPTIONS._make((2, 2,
                             2, 2, 2,
                              'train'))

    xgan=XGAN(options,encoder,decoder,discriminator,data)