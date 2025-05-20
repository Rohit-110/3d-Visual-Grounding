import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel
from pathlib import Path

from src.util.label import CLASS_LABELS_200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16', model_max_length=512, use_fast=True)

clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').cuda()
clip.share_memory()

class BaseConcept:
    def __init__(self, scan_data: dict, label_type=None) -> None:
        self.scan_data = scan_data
        self._init_params(label_type)
    
    def _init_params(self) -> None:
        raise NotImplementedError
    
    def forward(self) -> torch.Tensor:
        raise NotImplementedError

class CategoryConcept(BaseConcept):
    def _init_params(self, label_type) -> None:
        self.scan_id = self.scan_data['scan_id']
        if label_type == "gt":
            pred_class_list = self.scan_data['inst_labels']
        else:
            self.obj_embeds = self.scan_data['obj_embeds']
            self.class_name_list = list(CLASS_LABELS_200)
            self.class_name_list.remove('wall')
            self.class_name_list.remove('floor')
            self.class_name_list.remove('ceiling')

            self.class_name_tokens = tokenizer([f'a {class_name} in a scene.' for class_name in self.class_name_list],
                                                    padding=True,
                                                    return_tensors='pt')
            for name in self.class_name_tokens.data:
                self.class_name_tokens.data[name] = self.class_name_tokens.data[name].cuda()
            
            label_lang_infos = clip.get_text_features(**self.class_name_tokens)
            del self.class_name_tokens
            label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)
            class_logits_3d = torch.matmul(label_lang_infos, self.obj_embeds.t().cuda())     # * logit_scale
            obj_cls = class_logits_3d.argmax(dim=0)
            pred_class_list = [self.class_name_list[idx] for idx in obj_cls]
        new_class_list = pred_class_list
        new_class_name_tokens = tokenizer([f'a {class_name} in a scene.' for class_name in new_class_list],
                                               padding=True,
                                               return_tensors='pt')
        self.pred_class_list = pred_class_list
        self.obj_ids = self.scan_data['obj_ids']
        for name in new_class_name_tokens.data:
            new_class_name_tokens.data[name] = new_class_name_tokens.data[name].cuda()
        
        label_lang_infos = clip.get_text_features(**new_class_name_tokens)
        self.label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)
        
    @torch.inference_mode()
    def forward(self, category: str, color: str = None, shape: str = None) -> torch.Tensor:
        query_name_tokens = tokenizer([f'a {category} in a scene.'], padding=True, return_tensors='pt')
        for name in query_name_tokens.data:
            query_name_tokens.data[name] = query_name_tokens.data[name].cuda()
        query_lang_infos = clip.get_text_features(**query_name_tokens)
        query_lang_infos = query_lang_infos / query_lang_infos.norm(p=2, dim=-1, keepdim=True) # (768, )
        text_cls = torch.matmul(query_lang_infos, self.label_lang_infos.t()).squeeze() # (1, 768) * (768, N) -> (N, )
        text_cls = softmax(100 * text_cls, dim=0)
        return text_cls.to(DEVICE)

class ScanReferCategoryConcept(BaseConcept):
    def _init_params(self, label_type) -> None:
        self.scan_id = self.scan_data['scan_id']

        # --- NEW LOGIC ---
        # Check if 'pred_labels' is directly provided (it won't be by load_one_scene)
        if 'pred_labels' in self.scan_data:
             print(f"[ScanRefCatConcept] Using provided 'pred_labels' for {self.scan_id}")
             pred_class_list = self.scan_data['pred_labels']
        # If not, try to generate from 'obj_embeds' (provided by load_one_scene)
        elif 'obj_embeds' in self.scan_data and self.scan_data['obj_embeds'] is not None:
            print(f"[ScanRefCatConcept] Generating pred_labels from obj_embeds for {self.scan_id}...")
            obj_embeds = self.scan_data['obj_embeds'] # Should be shape [N_obj, Embed_Dim]
            if not isinstance(obj_embeds, torch.Tensor):
                 obj_embeds = torch.tensor(obj_embeds, dtype=torch.float32) # Ensure tensor

            # Ensure embeddings are on the correct device
            obj_embeds = obj_embeds.to(DEVICE)
            if obj_embeds.dim() == 1: # Handle case where only one object embedding might be loaded incorrectly
                 obj_embeds = obj_embeds.unsqueeze(0)

            # Generate text features for classification (similar to CategoryConcept)
            class_name_list = list(CLASS_LABELS_200) # Start with all possibilities
            # Remove ambiguous classes if desired
            for label in ['wall', 'floor', 'ceiling']:
                 if label in class_name_list: class_name_list.remove(label)

            # Prepare text prompts
            text_prompts = [f'a {class_name} in a scene.' for class_name in class_name_list]
            class_name_tokens = tokenizer(text_prompts, padding=True, return_tensors='pt')

            # Move tokens to device
            for name in class_name_tokens.data:
                class_name_tokens.data[name] = class_name_tokens.data[name].to(DEVICE)

            # Get text features from CLIP
            with torch.no_grad():
                 label_lang_infos = clip.get_text_features(**class_name_tokens)
            del class_name_tokens # Free memory
            label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True) # Normalize

            # Normalize object embeddings (assuming they aren't already)
            obj_embeds = obj_embeds / obj_embeds.norm(p=2, dim=-1, keepdim=True)

            # Calculate similarity scores (logits)
            # Shape: [Num_Classes, Embed_Dim] x [Embed_Dim, Num_Objects] -> [Num_Classes, Num_Objects]
            class_logits_3d = torch.matmul(label_lang_infos, obj_embeds.t())

            # Find the best class for each object
            obj_cls_indices = class_logits_3d.argmax(dim=0) # Shape: [Num_Objects]
            pred_class_list = [class_name_list[idx.item()] for idx in obj_cls_indices]
            print(f"[ScanRefCatConcept] Generated labels: {pred_class_list[:5]}...") # Print first few
        else:
             # If neither 'pred_labels' nor 'obj_embeds' are available
             print(f"ERROR in ScanReferCategoryConcept: Neither 'pred_labels' nor 'obj_embeds' found in scene_data for {self.scan_id}. Cannot determine labels.")
             # Fallback or raise error - Fallback to using 'object' for all might work but is inaccurate
             # Fallback: Use 'inst_labels' if available? Risky.
             if 'inst_labels' in self.scan_data and self.scan_data['inst_labels']:
                  print("WARN: Falling back to using 'inst_labels' as predicted labels.")
                  pred_class_list = self.scan_data['inst_labels']
             else:
                  raise ValueError(f"Missing required 'pred_labels' or 'obj_embeds' for {self.scan_id}")
        # --- END NEW LOGIC ---

        # --- The rest of the function proceeds using the determined pred_class_list ---
        if not pred_class_list:
            raise ValueError(f"pred_class_list is empty for {self.scan_id}")

        new_class_list = pred_class_list
        try:
            new_class_name_tokens = tokenizer([f'a {class_name} in a scene.' for class_name in new_class_list],
                                                padding=True,
                                                return_tensors='pt')
        except Exception as e:
             print(f"ERROR during tokenization (final) in ScanReferCategoryConcept for {self.scan_id}")
             print(f"Labels were: {new_class_list}")
             raise e

        self.pred_class_list = new_class_list # Store the labels actually used

        for name in new_class_name_tokens.data:
            new_class_name_tokens.data[name] = new_class_name_tokens.data[name].to(DEVICE)

        with torch.no_grad():
            label_lang_infos = clip.get_text_features(**new_class_name_tokens)

        del new_class_name_tokens # Free memory
        self.label_lang_infos = label_lang_infos / label_lang_infos.norm(p=2, dim=-1, keepdim=True)
    
    def forward(self, category: str, color: str = None, shape: str = None) -> torch.Tensor:
        query_name_tokens = tokenizer([f'a {category} in a scene.'], padding=True, return_tensors='pt')
        for name in query_name_tokens.data:
            query_name_tokens.data[name] = query_name_tokens.data[name].cuda()
        with torch.no_grad():
            query_lang_infos = clip.get_text_features(**query_name_tokens)
            query_lang_infos = query_lang_infos / query_lang_infos.norm(p=2, dim=-1, keepdim=True) # (768, )
            text_cls = torch.matmul(query_lang_infos, self.label_lang_infos.t()).squeeze() # (1, 768) * (768, N) -> (N, )
            text_cls = softmax(100 * text_cls, dim=0)     
            
        return text_cls.to(DEVICE)

class Near:
    def __init__(
        self, 
        object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object (x, y, z), 
                the last three columns are the size of the object (width, height, depth).
        """
        self.object_locations = object_locations.to(DEVICE)
        self._init_params()

    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Near` relation and initialize `self.param`.
        """
        # Based on average size to calculate a baseline distance
        sizes = self.object_locations[:, 3:]
        self.avg_size_norm = sizes.mean(dim=0).norm().to(DEVICE)

    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, N), where element (i, j) is the metric value of the `Near` relation between object i and object j.
        """
        centers = self.object_locations[:, :3]
        sizes = self.object_locations[:, 3:]
        
        # Calculate the pairwise distances between object centers in a vectorized manner
        diff = centers.unsqueeze(1) - centers.unsqueeze(0)
        distances = diff.norm(dim=2)
        
        # Calculate the "nearness" metric based on distances and average size norm
        nearness_metric = torch.exp(-distances / (self.avg_size_norm + 1e-6))
        
        # Set diagonal to zero since an object cannot be near itself
        nearness_metric.fill_diagonal_(0)
        
        return nearness_metric.to(DEVICE)

class Far:
    def __init__(
        self, 
        object_locations: torch.Tensor) -> None:
        """
        Args:
            object_locations: torch.Tensor, shape (N, 6), N is the number of objects in the scene.
                The first three columns are the center of the object (x, y, z), 
                the last three columns are the size of the object (width, height, depth).
        """
        self.object_locations = object_locations.to(DEVICE)
        self._init_params()

    def _init_params(self) -> None:
        """
        Computing some necessary parameters about `Far` relation and initialize `self.param`.
        """
        # Based on average size to calculate a baseline distance
        sizes = self.object_locations[:, 3:]
        self.avg_size_norm = sizes.mean(dim=0).norm().to(DEVICE)

    def forward(self) -> torch.Tensor:
        """
        Return a tensor of shape (N, N), where element (i, j) is the metric value of the `Far` relation between object i and object j.
        """
        centers = self.object_locations[:, :3]
        sizes = self.object_locations[:, 3:]
        
        # Calculate the pairwise distances between object centers in a vectorized manner
        diff = centers.unsqueeze(1) - centers.unsqueeze(0)
        distances = diff.norm(dim=2)
        
        # Calculate the "farness" metric based on distances and average size norm
        farness_metric = 1.0 - torch.exp(-distances / (self.avg_size_norm + 1e-6))
        
        # Set diagonal to zero since an object cannot be far from itself
        farness_metric.fill_diagonal_(0)
        
        return farness_metric.to(DEVICE)
    