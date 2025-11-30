import argparse
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.dataloader_sync import AudiosetDataset, unsupervised_collate_fn
from train_unsupervised import CAVMAEModule
import pandas as pd
from tqdm import tqdm
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
    
    return parser.parse_args()

class DeepFakeEvaluator(pl.LightningModule):
    
    def __init__(self, model_module):
        super().__init__()
        self.model_module = model_module
        self.results = []

    def forward(self, fbanks, images):
        return self.model_module(fbanks, images)

    def predict_step(self, batch, batch_idx):
        fbanks, images, video_ids, frame_indices = batch
        
        outputs = self(fbanks, images)
        
        loss = outputs['loss'].item()
        loss_mae = outputs['loss_mae'].item()
        loss_c = outputs['loss_c'].item()
        intra_acc = outputs['c_acc'].item()
        inter_acc = outputs['inter_acc'].item()
        
        # Assuming batch_size=1, all frames belong to the same video.
        # video_ids is a list of length (B*F).
        if len(video_ids) > 0:
            vid = video_ids[0]
        else:
            logging.warning(f"Empty video_ids in batch {batch_idx}")
            vid = "unknown"
        
        return {
            'video_id': vid,
            'loss': loss,
            'loss_mae': loss_mae,
            'loss_c': loss_c,
            'intra_acc': intra_acc,
            'inter_acc': inter_acc
        }

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = get_args()
    
    if args.batch_size != 1:
        logging.warning("batch_size is not 1. Scores will be averaged over the batch, which might not be useful for individual video detection.")
        logging.warning("Setting batch_size to 1 for evaluation.")
        args.batch_size = 1

    # Dataset Configuration
    audio_conf = {
        'num_mel_bins': args.num_mel_bins,
        'mean': args.mean,
        'std': args.std,
        'target_length': args.target_length,
        'mode': 'unsupervised_train', # Use this mode to get the same data loading behavior
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
        batch_size=args.batch_size, # Should be 1
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=unsupervised_collate_fn,
        pin_memory=True
    )
    
    logging.info(f"Loading checkpoint from {args.checkpoint_path}...")
    
    model = CAVMAEModule.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    evaluator = DeepFakeEvaluator(model)
    
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False
    )
    
    logging.info("Starting evaluation...")
    predictions = trainer.predict(evaluator, dataloader)
    
    # predictions is a list of dicts
    df = pd.DataFrame(predictions)
    
    logging.info(f"Saving results to {args.output_csv}...")
    df.to_csv(args.output_csv, index=False)
    logging.info("Done.")

if __name__ == "__main__":
    main()
