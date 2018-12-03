from model.data import Data
from model.discriminator import discriminator
from model.generator import encoder,decoder
from model.xgan import XGAN
from utils.data_io import show_img
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='horse2zebra', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--img_size', dest='img_size', type=int, default=128, help='# the size of the input image')
parser.add_argument('--ngf', dest='gf_dim', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='df_dim', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_c_dim', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_c_dim', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--is_training', dest='is_training', type=bool, default=False, help='is training or not')

args = parser.parse_args()

data=Data(args.img_size,is_train=args.is_training,test_img_path='./data/x_domain/Aaron_Eckhart_0001.jpg')
origin_img,_=data.next_batch(1)
xgan=XGAN(args,encoder,decoder,discriminator,data)
img,img1 = xgan.test('xgan20_d29.634479522705078_g15.58745563030243')
# show_img(data.next_batch(1)[0][0])

def uni(img):
    tmp = (img[0]+1)/2
    #tmp1=tmp-tmp.min()
    #tmp1=(tmp-tmp.min())/(tmp.max()-tmp.min())
    return tmp
show_img(origin_img[0])
show_img(img[0])
show_img(img1[0])

show_img(uni(origin_img))
show_img(uni(img))
show_img(uni(img1))