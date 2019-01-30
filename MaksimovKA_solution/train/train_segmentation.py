import gc
from params.params import args
from dataset.spacenet_binary_dataset import SpacenetBinaryDataset
from augmentations.transforms import augmentations
from train.model_factory import make_model
from keras.utils.training_utils import multi_gpu_model
from utils.losses import make_loss, binary_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from utils.metrics import hard_dice_coef_border, hard_jacard_coef_border, hard_dice_coef_mask, hard_jacard_coef_mask
from keras.optimizers import RMSprop, Adam, SGD
import keras.backend as K


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath,
                 monitor='val_loss',
                 verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


def main():
    folds = [int(f) for f in args.fold.split(",")]
    batch_size = args.batch_size
    if args.augment:
        transformer = augmentations()
    else:
        transformer = None
    optimizer_type = args.optimizer
    preprocessor = args.preprocessing_function
    models_dir = args.models_dir
    alias = args.alias
    network = args.network
    save_period = args.save_period
    multi_gpu = args.multi_gpu
    meta_weights_path = args.weights
    loss_function = args.loss_function
    logs = args.logs
    add_contours = args.add_contours
    freeze = args.freeze_encoder
    gpus = args.gpu.split(',')
    
    for fold in folds:
        
        if optimizer_type == 'rmsprop':
            optimizer = RMSprop(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'adam':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'amsgrad':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay), amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True, decay=float(args.decay))
            
        if multi_gpu:
            with K.tf.device("/cpu:0"):
                # model = make_model(args.network, (None, None, channels))
                model = make_model(args.network, freeze, 0)
        else:
            # model = make_model(args.network, (None, None, channels))
            model = make_model(args.network, freeze, 0)
        if args.weights is None:
            print('No weights passed, training from scratch')
        else:
            weights_path = meta_weights_path.format(fold)
            print('Loading weights from {}'.format(weights_path))
            model.load_weights(weights_path, by_name=True)
            
        dataset = SpacenetBinaryDataset(args.images_dir,
                                   args.masks_dir,
                                   args.folds_file,
                                   
                                   fold,
                                   args.n_folds,
                                   bool(add_contours))
    
        train_generator = dataset.train_generator(
            (args.crop_size, args.crop_size),
            preprocessor,
            transformer,
            batch_size=batch_size)
        
        val_generator = dataset.val_generator(preprocessor, batch_size=1)

        callbacks = []
        if bool(add_contours):
            metrics = [hard_dice_coef_mask,
                       hard_jacard_coef_mask,
                       hard_dice_coef_border,
                       hard_jacard_coef_border
                       ]
        else:
            metrics = [hard_dice_coef_mask,
                       hard_jacard_coef_mask,
                       binary_crossentropy]
        
        best_loss_model_file = '{}/best_loss_{}_{}{}_fold{}.h5'.format(models_dir, loss_function, alias, network, fold)
        
        best_loss_model = ModelCheckpointMGPU(model,
                                              filepath=best_loss_model_file,
                                              verbose=1,
                                              monitor='val_loss',
                                              mode='min',
                                              period=save_period,
                                              save_best_only=True,
                                              save_weights_only=True)
        callbacks.append(best_loss_model)

        def schedule_steps(epoch, steps):
            for step in steps:
                if step[1] > epoch:
                    print("Setting learning rate to {}".format(step[0]))
                    return step[0]
            print("Setting learning rate to {}".format(steps[-1][0]))
            return steps[-1][0]
        
        if args.schedule is not None:
            steps = [(float(step.split(":")[0]), int(step.split(":")[1])) for step in args.schedule.split(",")]
            print('Current steps will be perfomed')
            print(steps)
            lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, steps))
            callbacks.insert(0, lrSchedule)
            
        tb = TensorBoard("{}/{}_{}_{}".format(logs, network, fold, loss_function))
        callbacks.append(tb)

        use_reduce = args.use_reduce_lr
        if use_reduce:
            reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6,
                                          epsilon=0.001, verbose=1,
                                          mode='min')
            callbacks.append(reducer)
          
        if args.steps_per_epoch > 0:
            steps_per_epoch = args.steps_per_epoch
        else:
            steps_per_epoch = len(dataset.train_ids) / args.batch_size + 1
        #elif multi_gpu:
        #    steps_per_epoch = len(dataset.train_ids) / args.batch_size / len(gpus) + 1
        
        use_early = args.early_stopping
        if use_early:
            es = EarlyStopping(monitor='val_loss',  mode='min',  patience=args.early_patience)
            callbacks.append(es)

        if args.multi_gpu:
            model = multi_gpu_model(model, len(gpus))
        model.compile(loss=make_loss(loss_function),
                      optimizer=optimizer,
                      metrics=metrics)

        # model.summary()
        validation_data = val_generator
        validation_steps = len(dataset.val_ids)
        
        max_queue_size = args.max_queue_size
       
        verbose = args.verbose
        num_workers = args.num_workers
        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            verbose=verbose,
            workers=num_workers)

        del model
        K.clear_session()
        gc.collect()
        
        
if __name__ == '__main__':
    main()
