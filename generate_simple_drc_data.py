"""
Script to generate a simple DRC dataset with square shapes for verl training.
"""

import os
import shutil
import argparse
import pandas as pd
import gdsfactory as gf
import matplotlib.pyplot as plt

# 尝试导入我们之前定义的渲染工具，如果不存在则使用简易版
try:
    from verl.utils.drc.drc_tool import component_to_pil_image
except ImportError:
    print("Warning: ver.utils.drc.drc_tool not found. Using simple fallback renderer.")
    from matplotlib.patches import Polygon
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    import numpy as np

    def component_to_pil_image(component, title="layout", bbox=None):
        fig, ax = plt.subplots()
        for layer, polygons in component.get_polygons_points().items():
            for poly in polygons:
                ax.add_patch(Polygon(poly, closed=True, facecolor='blue', alpha=0.5))
        
        # 标注实例名
        if hasattr(component, "named_instances"):
            for name, ref in component.named_instances.items():
                cx, cy = ref.center
                ax.text(cx, cy, name, color="red", ha="center")

        ax.set_aspect('equal')
        ax.set_title(title)
        if bbox:
            ax.set_xlim(bbox[0], bbox[2])
            ax.set_ylim(bbox[1], bbox[3])
        else:
            ax.autoscale()
            
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        plt.close(fig)
        return img.convert("RGB")

def create_square_component(name: str, size: float = 10.0) -> gf.Component:
    """创建包含一个方块的组件"""
    c = gf.Component(name)
    
    # 创建一个矩形
    # layer (1, 0) 是常用的 GDS 图层
    rect = c.add_polygon(
        [(0, 0), (size, 0), (size, size), (0, size)], 
        layer=(1, 0)
    )
    
    # [重要] 必须添加为 Named Reference (实例)，因为我们的 Tool 是基于实例操作的
    # 为了简化，我们新建一个容器组件来引用这个矩形，或者直接给多边形打标（如果Tool支持）
    # 但根据之前的架构，Tool操作的是 Reference。
    # 所以标准做法是：
    top = gf.Component(f"{name}_top")
    ref = top.add_ref(c, name="square_1") # 实例名为 square_1
    ref.center = (0, 0)
    
    # 确保有 named_instances 属性 (兼容我们的 Tool)
    if not hasattr(top, "named_instances"):
        top.named_instances = {}
    top.named_instances["square_1"] = ref
    
    return top

def generate_dataset(output_dir: str, num_samples: int, split: str):
    """生成指定数量的样本并保存"""
    
    # 准备目录
    gds_dir = os.path.join(output_dir, "gds")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(gds_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    
    data_records = []
    
    print(f"Generating {num_samples} samples for split '{split}'...")
    
    for i in range(num_samples):
        sample_name = f"{split}_{i}"
        
        # 1. 创建几何
        # 这里可以加入随机性，比如随机大小或位置，让训练更有意义
        # size = 10.0 + (i % 5) 
        component = create_square_component(sample_name, size=10.0)
        
        # 2. 路径定义 (使用绝对路径，方便 verl 读取)
        gds_filename = f"{sample_name}.gds"
        png_filename = f"{sample_name}.png"
        
        abs_gds_path = os.path.abspath(os.path.join(gds_dir, gds_filename))
        abs_png_path = os.path.abspath(os.path.join(png_dir, png_filename))
        
        # 3. 保存文件
        component.write_gds(abs_gds_path)
        
        img = component_to_pil_image(component, title=sample_name)
        img.save(abs_png_path)
        
        # 4. 构造 Prompt
        # 这是 Generator Agent 看到的任务描述
        prompt = (
            "<image>\n"  # <--- 注意这里增加了 <image>
            "You have a clean layout with a square named 'square_1'. "
            "Your task is to create a DRC spacing violation. "
            "Use 'op_move_polygon' to move 'square_1' to a position where it might overlap or be too close to another shape if one existed, "
            "or simply demonstrate a move operation."
        )
        
        # 5. 记录元数据 (对应 DRCDataset 的字段)
        record = {
            "uid": sample_name,
            "data_source": "simple_square_drc",
            "split": split,
            "prompt": [{
                "role": "user",
                "content": prompt
            }],
            # "raw_prompt": [{
            #     "role": "user",
            #     "content": prompt
            # }],
            "images": [{"image": abs_png_path}],
            "clean_layout_gds_path": abs_gds_path,
            "clean_layout_png_path": abs_png_path,
            "target_drc_rule": "min_spacing", 
            "extra_info": {
    "interaction_kwargs": {
        "clean_layout_gds_path": abs_gds_path
    }
}
            # 可以在这里添加额外的字段供 Reward Function 使用
        }
        data_records.append(record)
        
    # 保存 Parquet
    df = pd.DataFrame(data_records)
    parquet_path = os.path.join(output_dir, f"{split}.parquet")
    df.to_parquet(parquet_path)
    print(f"Saved {len(df)} records to {parquet_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./drc_data_square", help="Output directory")
    parser.add_argument("--train_size", type=int, default=100, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=10, help="Number of validation samples")
    args = parser.parse_args()
    
    # 清理旧数据
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Cleaned up {args.output_dir}")
        
    generate_dataset(args.output_dir, args.train_size, "train")
    generate_dataset(args.output_dir, args.val_size, "val") # 注意 verl config 里的 val_files 名字要对应
    
    print("\nDataset generation complete!")
    print(f"Ensure your verl config points to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()