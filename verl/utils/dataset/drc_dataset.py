
import logging
from typing import Optional

import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)

class DRCDataset(RLHFDataset):
    """
    Dataset for the DRC generation and fixing task.
    Each item provides:
    - The initial "clean" layout as an image and a GDS file path.
    - A text prompt for the generator agent, describing the DRC error to create.
    """

    def __init__(
        self,
        data_files: str | list[str] | ListConfig,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # We will reuse the RLHFDataset's initialization for file handling
        # but override the item retrieval logic.
        super().__init__(data_files, tokenizer, config, processor, max_samples)

        # Ensure required columns exist
        required_columns = ["clean_layout_png_path", "clean_layout_gds_path", "generation_prompt"]
        for col in required_columns:
            if col not in self.dataframe.columns:
                raise ValueError(f"Required column '{col}' not found in the dataset.")

        print(f"DRC Dataset loaded with {len(self)} samples.")

    def __getitem__(self, item_index: int) -> dict:
        row = self.dataframe.iloc[item_index].to_dict()

        # Load the initial clean layout image
        try:
            image_path = copy_to_local(row["clean_layout_png_path"], use_shm=self.use_shm)
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image for index {item_index} at path {row['clean_layout_png_path']}: {e}")
            # Return a dummy image to avoid crashing the training loop
            image = Image.new('RGB', (224, 224), color = 'red')


        # The prompt for the generator agent
        # The prompt for the fixer agent will be created dynamically in the training loop
        generation_prompt = row["generation_prompt"]

        # The multi-modal data includes the initial image and paths to GDS files
        # The DRCInteraction will handle the GDS file loading.
        multi_modal_data = {
            "image": [image],
            "clean_gds_path": row["clean_layout_gds_path"],
        }
        
        # The `raw_prompt` will be a list of conversation turns. For the generator,
        # it starts with the initial instruction.
        raw_prompt = [{"role": "user", "content": generation_prompt}]

        # Other metadata needed by the reward function or interaction
        extra_info = {
            "target_drc_rule": row.get("target_drc_rule", "UNKNOWN"),
        }

        # The fixer will need a static prompt, which we define here for consistency
        fix_prompt = "You are a DRC fixing expert. The following layout has DRC violations. Please fix them using the provided tools."
        
        return {
            "raw_prompt": raw_prompt, # For generator
            "fix_prompt": fix_prompt, # For fixer
            "multi_modal_data": multi_modal_data,
            "extra_info": extra_info,
            "data_source": row.get("data_source", "drc_task"), # For reward function dispatch
            # Add a unique identifier for the sample
            "uid": row.get("uid", f"drc_sample_{item_index}")
        }

if __name__ == '__main__':
    # Simple test case for the dataset
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    import tempfile
    import os

    # Create dummy data and files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Dummy image
        dummy_img_path = os.path.join(tmpdir, "clean.png")
        Image.new('RGB', (100, 100)).save(dummy_img_path)

        # Dummy GDS file (just a text file for the test)
        dummy_gds_path = os.path.join(tmpdir, "clean.gds")
        with open(dummy_gds_path, "w") as f:
            f.write("GDS_DATA")

        # Dummy parquet file
        df = pd.DataFrame({
            "clean_layout_png_path": [dummy_img_path],
            "clean_layout_gds_path": [dummy_gds_path],
            "generation_prompt": ["Create a spacing violation between two polygons."],
            "target_drc_rule": ["MIN_SPACING"],
        })
        parquet_path = os.path.join(tmpdir, "drc_data.parquet")
        df.to_parquet(parquet_path)

        # Test config
        test_config = OmegaConf.create({
            "prompt_key": "generation_prompt",
            "use_shm": False,
        })
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        try:
            dataset = DRCDataset(parquet_files=[parquet_path], tokenizer=tokenizer, config=test_config)
            print(f"Dataset length: {len(dataset)}")
            sample = dataset[0]
            print("\nSample item:")
            for key, value in sample.items():
                if key == "multi_modal_data":
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {type(sub_value)}")
                else:
                    print(f"  {key}: {value}")
            assert "raw_prompt" in sample
            assert "fix_prompt" in sample
            assert "multi_modal_data" in sample
            assert "image" in sample["multi_modal_data"]
            print("\nDRCDataset test passed!")
        except Exception as e:
            print(f"\nDRCDataset test failed: {e}")

