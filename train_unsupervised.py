import argparse
import os
import logging
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from src.dataloader_sync import AudiosetDataset, unsupervised_collate_fn
from src.models.cav_mae_sync import CAVMAE

class CAVMAEModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # Handle args being a dict (when loading from checkpoint) or Namespace
        if isinstance(args, dict):
            from argparse import Namespace
            args = Namespace(**args)
        
        self.model = CAVMAE(
            img_size=args.im_res,
            audio_length=args.audio_length,
            embed_dim=args.embed_dim,
            total_frame=args.total_frame,
            contrastive_heads=args.contrastive_heads,
            cls_token=args.cls_token,
            num_register_tokens=args.num_register_tokens
        )

    def forward(self, fbanks, images, mode='unsupervised_train'):
        return self.model(
            fbanks, 
            images, 
            mask_ratio_a=self.hparams.mask_ratio_a, 
            mask_ratio_v=self.hparams.mask_ratio_v,
            mae_loss_weight=self.hparams.mae_loss_weight,
            contrast_loss_weight=self.hparams.contrast_loss_weight,
            mode=mode
        )

    def training_step(self, batch, batch_idx):
        fbanks, images, video_ids, frame_indices = batch
        
        outputs = self(fbanks, images)
        
        # Unpack outputs from dictionary
        loss = outputs['loss']
        loss_mae = outputs['loss_mae']
        loss_c = outputs['loss_c']
        intra_acc = outputs['c_acc']
        inter_acc = outputs['inter_acc']
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae_loss', loss_mae, on_step=True, on_epoch=True, logger=True)
        self.log('train_contrast_loss', loss_c, on_step=True, on_epoch=True, logger=True)
        self.log('train_intra_acc', intra_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_inter_acc', inter_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        fbanks, images, video_ids, frame_indices = batch
        
        outputs = self(fbanks, images)
        
        # Unpack outputs from dictionary
        loss = outputs['loss']
        loss_mae = outputs['loss_mae']
        loss_c = outputs['loss_c']
        intra_acc = outputs['c_acc']
        inter_acc = outputs['inter_acc']
        
        # Logging
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mae_loss', loss_mae, on_step=True, on_epoch=True, logger=True)
        self.log('test_contrast_loss', loss_c, on_step=True, on_epoch=True, logger=True)
        self.log('test_intra_acc', intra_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_inter_acc', inter_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

def get_args():
    parser = argparse.ArgumentParser(description="Train CAV-MAE Sync Unsupervised (Lightning)")
    
    # Dataset arguments
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--label_csv", type=str, default=None, help="Path to label CSV file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of videos)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--total_frame", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--target_length", type=int, default=416, help="Target audio length")
    parser.add_argument("--im_res", type=int, default=224, help="Image resolution")
    
    # Model arguments
    parser.add_argument("--audio_length", type=int, default=416, help="Audio length for model")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--mask_ratio_a", type=float, default=0.75, help="Audio mask ratio")
    parser.add_argument("--mask_ratio_v", type=float, default=0.75, help="Video mask ratio")
    parser.add_argument("--mae_loss_weight", type=float, default=1.0, help="Weight for MAE loss")
    parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="Weight for contrastive loss")
    parser.add_argument("--cls_token", action="store_true", help="Use CLS token")
    parser.add_argument("--num_register_tokens", type=int, default=8, help="Number of register tokens")
    parser.add_argument("--contrastive_heads", action="store_true", help="Use contrastive heads")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency (steps)")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint (ckpt file)")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run a quick development run")
    
    # Audio Conf defaults
    parser.add_argument("--num_mel_bins", type=int, default=128, help="Number of mel bins")
    parser.add_argument("--mean", type=float, default=-4.050048828125, help="Dataset mean")
    parser.add_argument("--std", type=float, default=4.067018032073975, help="Dataset std")
    
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    
    # Dataset Configuration
    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'mean': args.mean,
        'std': args.std,
        'target_length': args.target_length,
        'mode': 'unsupervised_train',
        'total_frame': args.total_frame,
        'im_res': args.im_res,
        'augmentation': True,
        'label_smooth': 0.0,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'custom',
        'skip_norm': False,
        'noise': False
    }

    logging.info("Audio Configuration:")
    logging.info(audio_conf)
    
    dataset = AudiosetDataset(
        dataset_json_file=args.dataset_json,
        audio_conf=audio_conf,
        label_csv=args.label_csv
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=unsupervised_collate_fn,
        pin_memory=True,  #  TODO: Check if this works
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    model = CAVMAEModule(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename='cav-mae-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        save_last=True,
        monitor='train_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices="auto" if torch.cuda.is_available() else 1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=args.log_freq,
        default_root_dir=args.save_path,
        fast_dev_run=args.fast_dev_run,
        # Resume training if checkpoint provided
    )
    
    ckpt_path = args.resume
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Checking checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location='cpu')
            # Heuristic to detect PL checkpoint
            if 'pytorch-lightning_version' in checkpoint or 'callbacks' in checkpoint:
                logging.info("Detected PyTorch Lightning checkpoint. Resuming training state...")
                ckpt_path = args.resume
            else:
                logging.info("Detected raw PyTorch checkpoint. Loading model weights only...")
                state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
                
                # Remove 'module.' prefix if present
                clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                # Load into appropriate part of the model
                if any(k.startswith('model.') for k in clean_state_dict):
                     msg = model.load_state_dict(clean_state_dict, strict=False)
                else:
                     msg = model.model.load_state_dict(clean_state_dict, strict=False)
                
                logging.info(f"Weights loaded manually. Message: {msg}")
                ckpt_path = None # Do not resume trainer state
                
        except Exception as e:
            logging.error(f"Failed to inspect checkpoint {args.resume}. Error: {e}")
            raise

    logging.info(f"Starting training with {len(dataset)} videos...")
    trainer.fit(model, dataloader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()