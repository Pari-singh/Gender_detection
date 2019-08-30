import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test  # import test.py to get mAP after each epoch
from models import *
from utils.adabound import *
from utils.datasets import *
from utils.utils import *

mixed_precision = True
from apex import amp


# Training hyperparameters g
# hyp = {'giou': 1.13,  # giou loss gain
#        'xy': 5.25,  # xy loss gain
#        'wh': 0.246,  # wh loss gain
#        'cls': 22.4,  # cls loss gain
#        'cls_pw': 3.64,  # cls BCELoss positive_weight
#        'obj': 17.8,  # obj loss gain
#        'obj_pw': 1.31,  # obj BCELoss positive_weight
#        'iou_t': 0.256,  # iou training threshold
#        'lr0': 0.00146,  # initial learning rate
#        'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.936,  # SGD momentum
#        'weight_decay': 0.00042,  # optimizer weight decay
#        'hsv_s': 0.123,  # image HSV-Saturation augmentation (fraction)
#        'hsv_v': 0.18,  # image HSV-Value augmentation (fraction)
#        'degrees': 1.81,  # image rotation (+/- deg)
#        'translate': 0.0987,  # image translation (+/- fraction)
#        'scale': 0.0788,  # image scale (+/- gain)
#        'shear': 0.441}  # image shear (+/- deg)


# Training hyperparameters i
hyp = {'giou': 1.43,  # giou loss gain
       'xy': 4.27,  # xy loss gain
       'wh': 0.265,  # wh loss gain
       'cls': 11.7,  # cls loss gain
       'cls_pw': 4.81,  # cls BCELoss positive_weight
       'obj': 11.5,  # obj loss gain
       'obj_pw': 1.56,  # obj BCELoss positive_weight
       'iou_t': 0.281,  # iou training threshold
       'lr0': 0.0013,  # initial learning rate
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.944,  # SGD momentum
       'weight_decay': 0.000427,  # optimizer weight decay
       'hsv_s': 0.0599,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.142,  # image HSV-Value augmentation (fraction)
       'degrees': 1.03,  # image rotation (+/- deg)
       'translate': 0.0552,  # image translation (+/- fraction)
       'scale': 0.0555,  # image scale (+/- gain)
       'shear': 0.434}  # image shear (+/- deg)

size = 416  # default value

def train(cfg,
          data,
          img_size=size,
          epochs=100,  # 500200 batches at bs 16, 117263 images = 273 epochs
          batch_size=16,
          accumulate=4):  # effective bs = batch_size * accumulate = 16 * 4 = 64
    # Initialize
    init_seeds()
    weights = 'weights' + os.sep
    last = weights + 'last.pt'
    best = weights + 'best.pt'
    device = torch_utils.select_device(apex=mixed_precision)
    multi_scale = opt.multi_scale

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'],
                          nesterov=True)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.
    if opt.resume or opt.transfer:  # Load previously saved model
        if opt.transfer:  # Transfer learning
            nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
            chkpt = torch.load(weights + 'yolov3.pt', map_location=device)
            model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                                  strict=False)

            for p in model.parameters():
                p.requires_grad = True if p.shape[0] == nf else False

        else:  # resume from last.pt
            if opt.bucket:
                os.system('gsutil cp gs://%s/last.pt %s' % (opt.bucket, last))  # download from bucket
            chkpt = torch.load(last, map_location=device)  # load checkpoint
            model.load_state_dict(chkpt['model'])

        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        if chkpt.get('training_results') is not None:
            with open('results.txt', 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    else:  # Initialize model with backbone (optional)
        if '-tiny.cfg' in cfg:
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        else:
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')

        # Remove old results
        for f in glob.glob('*_batch*.jpg') + glob.glob('results.txt'):
            os.remove(f)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in [0.8, 0.9]], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  image_weights=opt.img_weights)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=opt.num_workers,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    model.hyp = hyp  # attach hyperparameters to model
    if dataset.image_weights:
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    # n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n' + '%10s' * 9) %
              ('Epoch', 'gpu_mem', 'GIoU/xy', 'wh', 'obj', 'cls', 'total', 'targets', 'img_size'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = False
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(5).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            if multi_scale:
                if (i + nb * epoch) / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, paths=paths, fname='train_batch%g.jpg' % i)


            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, giou_loss=not opt.xywh)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 10)) or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test.test(cfg, data, batch_size=batch_size, img_size=opt.img_size, model=model,
                                          conf_thres=0.1)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Update best map
        fitness = results[2]  # mAP
        if fitness > best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt.nosave) or ((not opt.evolve) and (epoch == epochs - 1))
        if save:
            with open('results.txt', 'r') as file:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': file.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)
            if opt.bucket:
                os.system('gsutil cp %s gs://%s' % (last, opt.bucket))  # upload to bucket

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    # Report time
    print('%g epochs completed in %.3f hours.' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--accumulate', type=int, default=4, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='train at (1/1.5)x - 1.5x sizes')
    parser.add_argument('--img-size', type=int, default=size, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=2, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--xywh', action='store_true', help='use xywh loss instead of GIoU loss')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    opt = parser.parse_args()
    print(opt)

    if not opt.evolve:  # Train normally
        results = train(opt.cfg,
                        opt.data,
                        img_size=opt.img_size,
                        epochs=opt.epochs,
                        batch_size=opt.batch_size,
                        accumulate=opt.accumulate)

    else:  # Evolve hyperparameters (optional)
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(100):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Get best hyperparameters
                x = np.loadtxt('evolve.txt', ndmin=2)
                x = x[fitness(x).argmax()]  # select best fitness hyps
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = x[i + 5]

                # Mutate
                init_seeds(seed=int(time.time()))
                s = [.15, .15, .15, .15, .15, .15, .15, .15, .15, .00, .05, .20, .20, .20, .20, .20, .20, .20]  # sigmas
                for i, k in enumerate(hyp.keys()):
                    x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                    hyp[k] *= float(x)  # vary by sigmas

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale']
            limits = [(1e-4, 1e-2), (0.00, 0.70), (0.60, 0.97), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(opt.cfg,
                            opt.data,
                            img_size=opt.img_size,
                            epochs=opt.epochs,
                            batch_size=opt.batch_size,
                            accumulate=opt.accumulate)

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

