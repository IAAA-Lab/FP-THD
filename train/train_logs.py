import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial

def compute_loss(args, model, image, batch_size, criterion, text, length):
    print(f"[compute_loss] Input image shape: {image.shape}")  # Log shape of input
    preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    print(f"[compute_loss] Predictions shape: {preds.shape}")  # Log shape of predictions

    preds = preds.float()
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
    preds = preds.permute(1, 0, 2).log_softmax(2)
    
    # Check if there are NaNs in predictions
    if torch.any(torch.isnan(preds)):
        print("[compute_loss] WARNING: NaN values found in predictions!")

    torch.backends.cudnn.enabled = False
    loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True

    # Check if the loss is NaN
    if torch.isnan(loss):
        print("[compute_loss] WARNING: NaN loss detected!")
        print(f"Loss values: {loss}")
    return loss

def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    logger.info(f"1 Train data: {args.train_data_list}, Path: {args.data_path}, Image size: {args.img_size}")  # Log train dataset details
    logger.info(f"2 Train Dataset Ralph: {train_dataset.ralph}")  # Log the 'ralph' values from the dataset

    # Log the contents of 'ralph' to ensure it's correct
    #logger.info(f"ralph.values() preview: {list(train_dataset.ralph.values())[:5]}")  # Log a preview of the first few values
    logger.info(f" 3 ralph values: {list(train_dataset.ralph.values())}")

    character_list = list(train_dataset.ralph.values()) + ['ṕ']
    # Initialize CTCLabelConverter with ralph values
    converter = utils.CTCLabelConverter(character_list)

    logger.info(f"4 Initialized CTCLabelConverter with {len(train_dataset.ralph.values())} unique characters.")  # Log size of ralph values

    # Ensure the rest of the setup is working
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    character_list = list(train_dataset.ralph.values())+ ['З'] + ['Р'] #+ ['ṕ']
    print('character list added ', character_list)

    converter = utils.CTCLabelConverter(character_list)

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0

    #### ---- train & eval ---- ####

    for nb_iter in range(1, args.total_iter):

        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        print(f"[main] Train batch image shape: {image.shape}, Size: {image.size()}")  # Log shape and size of train batch image

        # Encode the text using the CTCLabelConverter
        text, length = converter.encode(batch[1])
        
        # Log the text encoding process to check for any issues
        logger.info(f"[main] Text to be encoded: {batch[1][:5]}")  # Log the first few text entries in the batch
        logger.info(f"[main] Encoded text: {text[:5]}")  # Log the first few encoded labels

        batch_size = image.size(0)
        loss = compute_loss(args, model, image, batch_size, criterion, text, length)

        # Log loss before backward pass
        if torch.isnan(loss):
            print(f"[main] NaN loss at iteration {nb_iter} before backward pass!")
        
        loss.backward()

        optimizer.first_step(zero_grad=True)
        compute_loss(args, model, image, batch_size, criterion, text, length).backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter

            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter)


                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                model.train()


if __name__ == '__main__':
    main()
