import cv2
import numpy as np
import torch
import threading
import torchreid


class ReIDFeatureExtractor:
    """
    Advanced Fast ReID feature extractor using OSNet-AIN x1.0
    (Omni-Scale Network with Attention Instance Normalization).
    
    This model explicitly learns cross-domain/cross-camera invariant features,
    making it much more robust against lighting and viewing angle changes.
    """

    def __init__(self, model_name='osnet_ain_x1_0', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            device=self.device
        )

        print(f"✅ Fast ReID loaded: {model_name} on {self.device}")

    @torch.no_grad()
    def extract(self, crop_bgr):
        """Extract a 512-dim L2-normalized feature vector from a BGR person crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        
        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 10:
            return None

        # Torchreid FeatureExtractor natively handles resizing, normalization, and tensor conversion
        # It expects RGB numpy arrays
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            # Native extractor output: (1, 512) tensor
            features = self.extractor([crop_rgb]).cpu().numpy()[0]
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            return features
        except Exception:
            return None

    @torch.no_grad()
    def extract_batch(self, crops_bgr):
        """Extract features for multiple crops in one forward pass."""
        valid_crops = []
        valid_indices = []

        for i, crop in enumerate(crops_bgr):
            if crop is not None and crop.size > 0:
                h, w = crop.shape[:2]
                if h >= 20 and w >= 10:
                    valid_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    valid_indices.append(i)

        results = [None] * len(crops_bgr)
        if not valid_crops:
            return results

        try:
            # Native extractor output: (N, 512) tensor
            features = self.extractor(valid_crops).cpu().numpy()
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            features = features / norms
            
            for idx, feat in zip(valid_indices, features):
                results[idx] = feat
        except Exception:
            pass

        return results
