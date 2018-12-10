
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tensorflow.contrib.layers as tcl
from model.generator import encoder,decoder,generator_resnet as generator
from model.discriminator import discriminator
from model.data import Data
from utils.utils import save_images
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

        #self.build_xgan()
        self.build_cycle()
    def build_xgan(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])

        # nets
        # G_part(originDomain_temDomain)
        self.domain_X_encode = self.encoder(self.X, self.options, reuse=False, name="encoderX")
        self.domain_Y_encode = self.encoder(self.Y, self.options, reuse=False, name="encoderY")

        self.domain_X_decode_Y = self.decoder( self.domain_X_encode, self.options, reuse=False,
                                              name="decoderY")
        self.domain_Y_decode_X = self.decoder( self.domain_Y_encode, self.options, reuse=False,
                                              name="decoderX")

        self.domain_X_encode_1 = self.encoder(self.domain_X_decode_Y, self.options, reuse=True, name="encoderY")
        self.domain_Y_encode_1 = self.encoder(self.domain_Y_decode_X, self.options, reuse=True, name="encoderX")

        self.domain_X_decode_X_1 = self.decoder( self.domain_X_encode_1, self.options, reuse=True,
                                              name="decoderX")
        self.domain_Y_decode_Y_1 = self.decoder( self.domain_Y_encode_1, self.options, reuse=True,
                                              name="decoderY")
        # tmp_feature1_1, self.domain_X_encode = self.encoder(self.X, self.options, reuse=False, name="encoderX")
        # tmp_feature2_1, self.domain_Y_encode = self.encoder(self.Y, self.options, reuse=False, name="encoderY")
        #
        # self.domain_X_decode_Y = self.decoder(tmp_feature1_1, self.domain_X_encode, self.options, reuse=False,
        #                                       name="decoderY")
        # self.domain_Y_decode_X = self.decoder(tmp_feature2_1, self.domain_Y_encode, self.options, reuse=False,
        #                                       name="decoderX")
        #
        # tmp_feature1_2, self.domain_X_encode_1 = self.encoder(self.domain_X_decode_Y, self.options, reuse=True,
        #                                                       name="encoderY")
        # tmp_feature2_2, self.domain_Y_encode_1 = self.encoder(self.domain_Y_decode_X, self.options, reuse=True,
        #                                                       name="encoderX")
        #
        # self.domain_X_decode_X_1 = self.decoder(tmp_feature1_2, self.domain_X_encode_1, self.options, reuse=True,
        #                                         name="decoderX")
        # self.domain_Y_decode_Y_1 = self.decoder(tmp_feature2_2, self.domain_Y_encode_1, self.options, reuse=True,
        #                                         name="decoderY")



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
        #self.classifier_score_X_real=self.classifier(self.domain_X_encode,reuse=False)
        #self.classifier_score_Y_real = self.classifier(self.domain_Y_encode,reuse=True)
        #self.classifier_score_X_fake = self.classifier(self.domain_Y_encode_1,reuse=True)
        #self.classifier_score_Y_fake = self.classifier(self.domain_X_encode_1,reuse=True)

        # loss 有问题  应该是不收敛 （G D 互相矛盾）
        # G_part
        #Lrec
        #换个损失函数
        # self.GXrealloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.domain_X_encode_1, labels=self.domain_X_encode))
        # self.GYrealloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.domain_Y_encode_1, labels=self.domain_Y_encode))
        # self.GYfakeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_fake, labels=tf.ones_like(self.DY_fake)))
        # self.GXfakeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_fake, labels=tf.ones_like(self.DX_fake)))
        # self.GXrealxloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_real_x, labels=tf.ones_like(self.DX_real_x)))
        # self.GYrealxloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_real_x, labels=tf.ones_like(self.DY_real_x)))
        self.GXrealloss = self.mae_criterion(self.domain_X_encode_1, self.domain_X_encode)
        self.GYrealloss = self.mae_criterion(self.domain_Y_encode_1, self.domain_Y_encode)
        self.GYfakeloss = self.mae_criterion(self.DY_fake, tf.ones_like(self.DY_fake))
        self.GXfakeloss = self.mae_criterion(self.DX_fake, tf.ones_like(self.DX_fake))

        #G_C_loss先去掉
        #self.G_CXloss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_X_fake,
        #                                                                   labels=tf.ones_like(self.classifier_score_X_fake)))
        #self.G_CYloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_Y_fake,
        #                                                                     labels=tf.ones_like(self.classifier_score_Y_fake)))
        self.G_loss = self.GXrealloss + self.GYrealloss + \
                      0.5*self.GYfakeloss + 0.5*self.GXfakeloss \
                      #+ \
                      #0.5 *self.GXrealxloss + 0.5*self.GYrealxloss\
                      #+\
                      #0.5*self.G_CXloss + 0.5*self.G_CYloss
        # D_part
        self.D_Xloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_fake, labels=tf.zeros_like(self.DX_fake)))+\
                       tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DX_real, labels=tf.ones_like(self.DX_real)))

        self.D_Yloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_fake, labels=tf.zeros_like(self.DY_fake))) + \
                       tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.DY_real, labels=tf.ones_like(self.DY_real)))
        #D_C_loss
        # self.D_CXloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_X_real,
        #                                                                             labels=tf.ones_like(self.classifier_score_X_real)))
        # self.D_CXloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_X_fake,
        #                                                                             labels=tf.zeros_like(self.classifier_score_X_fake)))
        # self.D_CYloss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_Y_real,
        #                                                                             labels=tf.ones_like(self.classifier_score_Y_real)))
        # self.D_CYloss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.classifier_score_Y_fake,
        #                                                                             labels=tf.zeros_like(self.classifier_score_Y_fake)))

        self.D_loss = self.D_Xloss + self.D_Yloss
                      #+\
            #0.5*self.D_CXloss_real + 0.5*self.D_CXloss_fake +\
            #0.5*self.D_CYloss_real + 0.5*self.D_CYloss_fake

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'encoder' in var.name or 'decoder' in var.name]

        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss,
                                                                                       var_list=self.d_vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss,
                                                                     var_list=self.g_vars)
    def build_cycle(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])

        self.fake_B = generator(self.X, self.options, False, name="generatorA2B")
        self.fake_A_ = generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = generator(self.Y, self.options, True, name="generatorB2A")
        self.fake_B_ = generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        # 此处discriminator参数更新了吧？ 并没有更新～在tf.train.AdamOptimizer中var_list指定更新的参数
        # 测试loss 改变(关系不大)
        self.g_loss_a2b = self.mae_criterion(self.DB_fake, tf.ones_like(self.DB_fake)) \
                          +  self.abs_criterion(self.X, self.fake_A_) \
                          +  self.abs_criterion(self.Y, self.fake_B_)
        self.g_loss_b2a = self.mae_criterion(self.DA_fake, tf.ones_like(self.DA_fake)) \
                          +  self.abs_criterion(self.X, self.fake_A_) \
                          + self.abs_criterion(self.Y, self.fake_B_)
        self.G_loss = self.mae_criterion(self.DA_fake, tf.ones_like(self.DA_fake)) \
                      + self.mae_criterion(self.DB_fake, tf.ones_like(self.DB_fake)) \
                      +  self.abs_criterion(self.X, self.fake_A_) \
                      + self.abs_criterion(self.Y, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.size, self.size,
                                             self.options.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.size, self.size,
                                             self.options.input_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.Y, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.X, self.options, reuse=True, name="discriminatorA")
        #self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        #self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.mae_criterion(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.mae_criterion(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.mae_criterion(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.mae_criterion(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.D_loss = self.da_loss + self.db_loss
        '''
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")
        '''
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        #for var in t_vars: print(var.name)
        self.D_solver = tf.train.AdamOptimizer(0.0002, beta1=0.5) \
            .minimize(self.D_loss, var_list=self.d_vars)
        self.G_solver = tf.train.AdamOptimizer(0.0002, beta1=0.5) \
            .minimize(self.G_loss, var_list=self.g_vars)
    def mae_criterion(self,in_, target):
        return tf.reduce_mean((in_-target)**2)

    def abs_criterion(self,in_, target):
        return tf.reduce_mean(tf.abs(in_ - target))

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
        # clip
        #tmp1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminatorYdomain')
        #tmp2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminatorXdomain')
        #tmp_vars = tmp1+tmp2
        #self.clip_D = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in tmp_vars]

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
                #self.sess.run(self.clip_D)
                X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)

                #_ , img =self.sess.run(
                #   [self.D_solver,self.domain_X_decode_Y],
                #   feed_dict={self.X: X_domain_image, self.Y: Y_domain_image})
                _, img = self.sess.run(
                    [self.D_solver, self.fake_B],
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
                    if i%100 ==0:
                        #Y_domain_image_ = self.sess.run(self.domain_X_decode_Y,
                         #                               feed_dict={self.X: X_domain_image_val[i][np.newaxis,:,:,:],
                          #                                         self.Y: Y_domain_image_val[i][np.newaxis,:,:,:]
                           #                                        }
                        Y_domain_image_ = self.sess.run(self.fake_B,
                                                        feed_dict={self.X: X_domain_image_val[i][np.newaxis, :, :, :],
                                                                   self.Y: Y_domain_image_val[i][np.newaxis, :, :, :]
                                                                   }
                                                        )
                        save_images(Y_domain_image_, [1, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format('train_output', epoch, i))
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
            if epoch in [2,5,10,20,50,100,150,200]:
                self.saver.save(self.sess, os.path.join(os.getcwd(),'model_save', 'xgan%s_d%s_g%s.ckpt'%(epoch,D_loss_curr,G_loss_curr)))
                print('model save at %s'%os.path.join(os.getcwd(),'model_save', 'xgan.ckpt'))

    def test(self,model_name):
        # # with tf.Graph().as_default():#加了就出问题。。。。？？？？？！！！！！！
        # self.saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'model_save' , (model_name+'.ckpt.meta')))
        # gpu_options = tf.GPUOptions(allow_growth=True)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # #self.sess.run(tf.global_variables_initializer())
        # self.saver.restore(self.sess,os.path.join(os.getcwd(), 'model_save' , (model_name+'.ckpt')))
        # X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
        # #Y_domain_image_,X_domain_image_ = self.sess.run([self.domain_X_decode_Y,self.domain_X_decode_X_1],
        #  #                               feed_dict={self.X: X_domain_image,
        #   #                                         self.Y: Y_domain_image}
        #    #                             )
        # Y_domain_image_, X_domain_image_ = self.sess.run([self.fake_B, self.fake_A_],
        #                                                  feed_dict={self.X: X_domain_image,
        #                                                             self.Y: Y_domain_image}
        #                                                  )
        # return Y_domain_image_,X_domain_image_
        self.saver = tf.train.Saver()
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            #model_dir = "%s_%s" % (model_name)
            checkpoint_dir = os.path.join('model_save', model_name)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(sess, os.path.join(os.getcwd(),checkpoint_dir, ckpt_name))
                X_domain_image, Y_domain_image = self.data.next_batch(self.options.batch_size)
                Y_domain_image_, X_domain_image_ = sess.run([self.fake_B, self.fake_A_],
                                                feed_dict={self.X: X_domain_image,
                                                            self.Y: Y_domain_image}
                                                )
                return Y_domain_image_, X_domain_image_
            else:
                print('error')



if __name__=='__main__':
    from collections import namedtuple

    data=Data()
    OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                                  gf_dim df_dim output_c_dim is_training')
    options = OPTIONS._make((2, 2,
                             2, 2, 2,
                              'train'))

    xgan=XGAN(options,encoder,decoder,discriminator,data)