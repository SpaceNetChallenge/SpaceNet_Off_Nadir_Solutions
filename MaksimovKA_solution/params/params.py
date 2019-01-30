import argparse
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--schedule')
arg('--seed', type=int, default=769)
arg('--gpu', default="0")
arg('--fold', default='0')
arg('--epochs', type=int, default=100)
arg('--training_data', default=' /data/SpaceNet-Off-Nadir_Train')
arg('--output_data', default='/wdata')
arg('--folds_file', default='/wdata/folds_split.csv')
arg('--n_folds', type=int, default=5)
arg('--augment', type=int, default=1)
arg('--images_dir', default='/wdata/train/images')
arg('--masks_dir', default='/wdata/train/masks')
arg('--add_contours', type=int, default=1)

arg('--default_size', type=int, default=928)
arg('--predict_size', type=int, default=900)

arg('--crop_size', type=int, default=320)
arg('--batch_size', type=int, default=64)
arg('--network', default='inceptionresnet_unet_borders')
arg('--preprocessing_function', type=int, default=0)
arg('--freeze_encoder', type=int, default=0)
arg('--alias', default='')
arg('--save_period', type=int, default=1)
arg('--steps_per_epoch', type=int, default=0)
arg('--max_queue_size', type=int, default=10)
arg('--verbose', type=int, default=1)
arg('--num_workers', type=int, default=20)
arg('--optimizer', default="adam")
arg('--learning_rate', type=float, default=1e-3)
arg('--decay', type=float, default=0.0)
arg('--weights')
arg('--loss_function', default='double_head')
arg('--multi_gpu', action="store_true")
arg('--logs', default="/wdata/models_logs")
arg('--use_reduce_lr', type=int, default=1)
arg('--early_stopping', type=int, default=1)
arg('--early_patience', type=int, default=8)

arg('--models_dir', default='/wdata/models_weights')
arg('--models', nargs='+', default=['best_loss_double_head_inceptionresnet_unet_borders_fold0.h5',
									'best_loss_double_head_inceptionresnet_unet_borders_fold1.h5',
									'best_loss_double_head_inceptionresnet_unet_borders_fold2.h5',
									'best_loss_double_head_inceptionresnet_unet_borders_fold3.h5',
									'best_loss_double_head_inceptionresnet_unet_borders_fold4.h5',


									'best_loss_double_head_inceptionresnet_fpn_borders_fold0.h5',
									'best_loss_double_head_inceptionresnet_fpn_borders_fold1.h5',
									'best_loss_double_head_inceptionresnet_fpn_borders_fold2.h5',
									'best_loss_double_head_inceptionresnet_fpn_borders_fold3.h5',
									'best_loss_double_head_inceptionresnet_fpn_borders_fold4.h5',
									])

arg('--test_folder', default='/data/SpaceNet-Off-Nadir_Test')
arg('--submit_output_file', default='/wdata/submit.txt')
arg('--out_root_dir', default='/wdata')
arg('--out_masks_folder', default='test_predictions')


args = parser.parse_args()

