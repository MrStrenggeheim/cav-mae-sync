import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataloader_sync import AudiosetDataset, unsupervised_collate_fn
from train_unsupervised import CAVMAEModule
import pandas as pd
import logging

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate CAV-MAE Sync for DeepFake Detection")
    
    # Dataset arguments
    parser.add_argument("--dataset_json", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_csv", type=str, default="eval_results.csv", help="Path to save results CSV")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of videos)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--total_frame", type=int, default=16, help="Number of frames per video")
    parser.add_argument("--target_length", type=int, default=416, help="Target audio length")
    parser.add_argument("--im_res", type=int, default=224, help="Image resolution")
    
    # Audio config (should match training)
    parser.add_argument("--num_mel_bins", type=int, default=128, help="Number of mel bins")
    parser.add_argument("--mean", type=float, default=-5.081, help="Dataset mean")
    parser.add_argument("--std", type=float, default=4.4849, help="Dataset std")
    
    # Evaluation config
    parser.add_argument("--use_masking", action="store_true", default=True, 
                        help="Use default masking ratios from checkpoint (default: True)")
    parser.add_argument("--no_masking", dest="use_masking", action="store_false",
                        help="Disable masking (set mask_ratio to 0)")
    
    return parser.parse_args()

class DeepFakeEvaluator(pl.LightningModule):
    
    def __init__(self, model_module, use_masking=True):
        super().__init__()
        self.model_module = model_module
        self.use_masking = use_masking
        self.test_outputs = []

    def forward(self, fbanks, images):
        if self.use_masking:
            # Use default mask ratios from checkpoint
            return self.model_module(fbanks, images)
        else:
            # No masking - call underlying model directly with mask_ratio=0
            return self.model_module.model(
                fbanks, images,
                mask_ratio_a=0.0,
                mask_ratio_v=0.0,
                mae_loss_weight=self.model_module.hparams.mae_loss_weight,
                contrast_loss_weight=self.model_module.hparams.contrast_loss_weight,
                mode='unsupervised_train'
            )

    def test_step(self, batch, batch_idx):
        fbanks, images, video_ids, frame_indices = batch
        
        outputs = self(fbanks, images)
        
        loss = outputs['loss']
        loss_mae = outputs['loss_mae']
        loss_c = outputs['loss_c']
        intra_acc = outputs['c_acc']
        inter_acc = outputs['inter_acc']
        
        # Log metrics - Lightning handles step/epoch aggregation
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('eval_mae_loss', loss_mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('eval_contrast_loss', loss_c, on_step=True, on_epoch=True, batch_size=1)
        self.log('eval_intra_acc', intra_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('eval_inter_acc', inter_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        
        vid = video_ids[0] if len(video_ids) > 0 else "unknown"
        
        result = {
            'video_id': vid,
            'loss': loss.item(),
            'loss_mae': loss_mae.item(),
            'loss_c': loss_c.item(),
            'intra_acc': intra_acc.item(),
            'inter_acc': inter_acc.item()
        }
        self.test_outputs.append(result)
        return result

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    
    if args.batch_size != 1:
        logging.warning("Setting batch_size to 1 for per-video evaluation.")
        args.batch_size = 1

    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'mean': args.mean,
        'std': args.std,
        'target_length': args.target_length,
        'mode': 'unsupervised_train',
        'total_frame': args.total_frame,
        'im_res': args.im_res,
        'augmentation': False,
        'label_smooth': 0.0,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'custom',
        'skip_norm': False,
        'noise': False
    }
    
    dataset = AudiosetDataset(
        dataset_json_file=args.dataset_json,
        audio_conf=audio_conf,
        label_csv=None
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=unsupervised_collate_fn,
        pin_memory=True
    )
    
    logging.info(f"Loading checkpoint from {args.checkpoint_path}...")
    model = CAVMAEModule.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    logging.info(f"Using masking: {args.use_masking}")
    evaluator = DeepFakeEvaluator(model, use_masking=args.use_masking)
    
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto',
        enable_checkpointing=False,
        log_every_n_steps=10
    )
    
    logging.info(f"Evaluating {len(dataset)} samples...")
    trainer.test(evaluator, dataloader)
    
    df = pd.DataFrame(evaluator.test_outputs)
    df.to_csv(args.output_csv, index=False)
    logging.info(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
