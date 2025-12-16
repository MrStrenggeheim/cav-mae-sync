from dataclasses import dataclass, field
from typing import Optional
import logging


@dataclass
class FakeSyncConfig:
    
    # ===================
    # Model Architecture
    # ===================
    img_size: int = 224
    audio_length: int = 48  # MUST be divisible by 16 (patch size)
    embed_dim: int = 768
    num_register_tokens: int = 8
    cls_token: bool = True
    contrastive_heads: bool = False
    
    # ===================
    # Data Configuration
    # ===================
    batch_size: int = 12  # Number of videos per batch
    total_frame: int = 16  # Frames sampled per video
    target_length: int = 48  # Target fbank length (MUST equal audio_length)
    im_res: int = 224  # Image resolution
    
    # ===================
    # Loss Configuration
    # ===================
    mask_ratio_a: float = 0.75  # Audio mask ratio
    mask_ratio_v: float = 0.75  # Video mask ratio
    mae_loss_weight: float = 1.0
    contrast_loss_weight: float = 0.1
    
    # ===================
    # Optimizer
    # ===================
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    epochs: int = 10
    
    # ===================
    # Audio Processing
    # ===================
    num_mel_bins: int = 128
    mean: float = -4.050048828125  # VoxCeleb2 normalization mean
    std: float = 4.067018032073975  # VoxCeleb2 normalization std
    
    # ===================
    # Training Logistics
    # ===================
    log_freq: int = 100  # TensorBoard logging interval (steps)
    checkpoint_interval_hours: float = 1.0
    
    # ===================
    # Paths (set at runtime)
    # ===================
    sharded_dataset_dir: Optional[str] = None
    val_shard_dir: Optional[str] = None  # Future: validation dataset
    save_path: str = "./checkpoints"
    resume: Optional[str] = None
    dataset_json: Optional[str] = None  # Legacy: non-sharded dataset
    label_csv: Optional[str] = None  # Legacy: label CSV file
    num_workers: int = 4  # DataLoader workers per GPU
    fast_dev_run: bool = False  # Quick test with 1 batch
    
    # ===================
    # Contrastive Loss
    # ===================
    contrast_bidirect: bool = False
    contrast_intra_weight: float = 0.6
    contrast_inter_weight: float = 0.4
    
    # ===================
    # Sync Score (Evaluation)
    # ===================
    sync_similarity: str = 'cosine'  # 'cosine' or 'softmax'
    sync_temperature: float = 0.05  # Temperature for softmax similarity
    sync_aggregation: str = 'p10'  # 'mean', 'min', 'p10', 'p25'
    sync_temporal_variance: bool = True  # Also compute variance across frames
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid or inconsistent.
        """
        errors = []
        
        # Audio length MUST be divisible by 16 (patch size)
        if self.audio_length % 16 != 0:
            errors.append(
                f"audio_length ({self.audio_length}) must be divisible by 16. "
                f"Valid values: 48, 64, 80, 96, 112, 128, ..."
            )
        
        # target_length must match audio_length
        if self.target_length != self.audio_length:
            errors.append(
                f"target_length ({self.target_length}) must equal "
                f"audio_length ({self.audio_length}) for correct model behavior"
            )
        
        # Batch size must be positive
        if self.batch_size < 1:
            errors.append(f"batch_size ({self.batch_size}) must be >= 1")
        
        # Mask ratios must be in valid range
        for name, val in [("mask_ratio_a", self.mask_ratio_a), 
                          ("mask_ratio_v", self.mask_ratio_v)]:
            if not 0.0 <= val <= 1.0:
                errors.append(f"{name} ({val}) must be in [0.0, 1.0]")
        
        # Epochs and warmup must be non-negative
        if self.epochs < 1:
            errors.append(f"epochs ({self.epochs}) must be >= 1")
        if self.warmup_epochs < 0:
            errors.append(f"warmup_epochs ({self.warmup_epochs}) must be >= 0")
        if self.warmup_epochs >= self.epochs:
            errors.append(
                f"warmup_epochs ({self.warmup_epochs}) must be < epochs ({self.epochs})"
            )
        
        # Learning rate must be positive
        if self.lr <= 0:
            errors.append(f"lr ({self.lr}) must be > 0")
        
        # total_frame must be positive
        if self.total_frame < 1:
            errors.append(f"total_frame ({self.total_frame}) must be >= 1")
        
        # Image resolution must be positive and reasonable
        if self.im_res < 32 or self.im_res > 1024:
            errors.append(f"im_res ({self.im_res}) should be between 32 and 1024")
        
        # Embed dim must be divisible by num_heads (12 is default)
        if self.embed_dim % 12 != 0:
            errors.append(
                f"embed_dim ({self.embed_dim}) must be divisible by 12 (num_heads)"
            )
        
        # Sync score parameters
        valid_similarity = ('cosine', 'softmax')
        if self.sync_similarity not in valid_similarity:
            errors.append(
                f"sync_similarity ({self.sync_similarity}) must be one of {valid_similarity}"
            )
        valid_aggregation = ('mean', 'min', 'p10', 'p25')
        if self.sync_aggregation not in valid_aggregation:
            errors.append(
                f"sync_aggregation ({self.sync_aggregation}) must be one of {valid_aggregation}"
            )
        
        if errors:
            error_msg = "FakeSyncConfig validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)
        
        logging.debug("FakeSyncConfig validation passed")
    
    @classmethod
    def from_args(cls, args) -> "FakeSyncConfig":
        """
        Create a FakeSyncConfig from an argparse Namespace.
        
        Only copies fields that exist in both the Namespace and the dataclass.
        
        Args:
            args: argparse.Namespace or dict with configuration values
            
        Returns:
            FakeSyncConfig instance with values from args
        """
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = dict(args)
        
        # Only include fields that are defined in this dataclass
        valid_fields = {f for f in cls.__dataclass_fields__}
        # Print a warning for any unknown fields
        unknown_fields = set(args_dict) - valid_fields
        if unknown_fields:
            logging.warning(f"Unknown fields in args encountered during config loading: {unknown_fields}")
        filtered = {k: v for k, v in args_dict.items() if k in valid_fields}
        
        return cls(**filtered)
    
    def to_audio_conf(self) -> dict:
        """
        Generate audio_conf dict for dataloader.
        
        Returns:
            Dict compatible with ShardedAudiosetDataset and AudiosetDataset
        """
        return {
            'num_mel_bins': self.num_mel_bins,
            'mean': self.mean,
            'std': self.std,
            'target_length': self.target_length,
            'mode': 'unsupervised_train',
            'total_frame': self.total_frame,
            'im_res': self.im_res,
            'augmentation': True,
            'label_smooth': 0.0,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': 'custom',
            'skip_norm': False,
            'noise': False
        }
