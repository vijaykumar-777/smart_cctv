import cv2
import numpy as np
import torch
import threading
import torchreid


class ReIDFeatureExtractor:
    """
    Fast ReID feature extractor using OSNet (Omni-Scale Network).
    
    OSNet is specifically designed for person re-identification with
    multi-scale feature learning. The osnet_x0_25 variant is lightweight
    (~200K params) while maintaining strong ReID accuracy.
    
    Produces 512-dim L2-normalized feature embeddings per person crop.
    """

    def __init__(self, model_name='osnet_x1_0', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load pre-trained OSNet model (trained on ImageNet, fine-tuned for ReID)
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            device=self.device
        )

        # Standard ReID preprocessing
        self.input_size = (256, 128)  # height x width (ReID convention)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print(f"✅ Fast ReID model loaded: {model_name} on {self.device}")

    @torch.no_grad()
    def extract(self, crop_bgr):
        """
        Extract a 512-dim L2-normalized feature vector from a BGR person crop.
        
        Args:
            crop_bgr: BGR image crop (numpy array) of a detected person
            
        Returns:
            512-dim numpy feature vector, or None if crop is invalid
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 5:
            return None

        try:
            # Convert BGR -> RGB and resize to ReID input
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (self.input_size[1], self.input_size[0]))

            # Normalize to [0, 1] then apply ImageNet normalization
            img = crop_resized.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std

            # Convert to tensor: HWC -> CHW -> NCHW
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

            # Extract features
            features = self.extractor.model(tensor)
            features = features.squeeze().cpu().numpy()

            # L2 normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features
        except Exception as e:
            return None

    @torch.no_grad()
    def extract_batch(self, crops_bgr):
        """
        Extract features for multiple crops in a single batch (faster).
        
        Args:
            crops_bgr: List of BGR image crops
            
        Returns:
            List of 512-dim feature vectors (None for invalid crops)
        """
        valid_indices = []
        tensors = []

        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
                continue
            try:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_resized = cv2.resize(crop_rgb, (self.input_size[1], self.input_size[0]))
                img = crop_resized.astype(np.float32) / 255.0
                img = (img - self.mean) / self.std
                tensors.append(torch.from_numpy(img.transpose(2, 0, 1)))
                valid_indices.append(i)
            except Exception:
                continue

        results = [None] * len(crops_bgr)

        if not tensors:
            return results

        try:
            batch = torch.stack(tensors).to(self.device)
            features = self.extractor.model(batch)
            features = features.cpu().numpy()

            # L2 normalize each feature vector
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1
            features = features / norms

            for idx, feat in zip(valid_indices, features):
                results[idx] = feat
        except Exception:
            pass

        return results
