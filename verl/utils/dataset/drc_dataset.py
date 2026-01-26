# verl/utils/dataset/drc_dataset.py

import logging
from typing import Optional
import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.fs import copy_to_local

class DRCDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples=-1):
        super().__init__(data_files, tokenizer, config, processor, max_samples)

    def __getitem__(self, item_index: int) -> dict:
        row = self.dataframe[item_index]

        # 1. 加载图片
        try:
            image_path = copy_to_local(row["clean_layout_png_path"], use_shm=self.use_shm)
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (224, 224), color='red')

        # 2. [简化] 不需要手动拼接 System Prompt
        # 直接返回用户的原始指令，verl 会自动处理工具定义
        raw_prompt = [
            {"role": "user", "content": row["generation_prompt"]}
        ]
        
        # Fixer 的 Prompt 同理
        fix_prompt = [
             {"role": "user", "content": "The previous layout had errors. Please fix them."}
        ]

        return {
            "raw_prompt": raw_prompt,
            "fix_prompt": fix_prompt,
            "multi_modal_data": {
                "image": [image],
                "clean_gds_path": row["clean_layout_gds_path"],
            },
            "extra_info": {
                "target_drc_rule": row.get("target_drc_rule", "UNKNOWN"),
            },
            "data_source": row.get("data_source", "drc_task"),
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

  
        dataset = DRCDataset(data_files=[parquet_path], tokenizer=tokenizer, config=test_config)
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
        

