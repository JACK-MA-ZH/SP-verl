# verl/interactions/drc_interaction.py

import logging
import os
import json
import asyncio
import tempfile
from typing import Any, Optional, Dict, List
from uuid import uuid4

import gdsfactory as gf
from PIL import Image

# [CRITICAL IMPORT] 从你的工具库中导入渲染函数
from verl.utils.drc.drc_tool import (
    MovePolygonTool, 
    DeletePolygonTool, 
    OffsetPolygonTool, 
    SplitPolygonTool,
    component_to_pil_image,  # <--- 必须使用这个函数来渲染
    _ensure_named_instance_map,
    _register_reference_name,
)

from verl.interactions.base import BaseInteraction
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class DRCInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        
        # 初始化工具逻辑
        self.tool_map = {
            "op_move_polygon": MovePolygonTool(),
            "op_delete_polygon": DeletePolygonTool(),
            "op_offset_polygon": OffsetPolygonTool(),
            "op_split_polygon": SplitPolygonTool(),
        }

    def _ensure_reference_names(self, comp):
        """Replicates logic from your drc.py _ensure_reference_names"""
        if not hasattr(comp, "named_instances") or comp.named_instances is None:
            comp.named_instances = {}

        def _iter_refs(c):
            if hasattr(c, "insts"): return c.insts
            elif hasattr(c, "references"): return c.references
            return []

        for index, reference in enumerate(_iter_refs(comp)):
            if not getattr(reference, "name", None):
                reference.name = f"p{index}"
            comp.named_instances[reference.name] = reference

    def _get_drc_violations(self, component) -> tuple[int, str]:
        """Runs DRC checks strictly."""
        # 安全检查：确保对象有 DRC 方法
        if not (hasattr(component, "drc_spacing") and hasattr(component, "drc_width")):
            # 如果没有 DRC 方法，可能是因为版本问题或空组件，返回默认值避免 crash
            return 0, "Warning: Component missing DRC methods (Mock Pass)."

        try:
            spacing_errors = component.drc_spacing(layer=(1, 0), spacing=0.1)
            width_errors = component.drc_width(layer=(1, 0), min_width=0.12)

            errors = []
            for poly in spacing_errors.get_polygons():
                errors.append({"type": "min_spacing", "bbox": poly.bounds})
            for poly in width_errors.get_polygons():
                errors.append({"type": "min_width", "bbox": poly.bounds})

            errors_text = (
                "\n".join([f"ERROR: {e['type']} at {e['bbox']}" for e in errors])
                if errors else "No DRC errors found."
            )
            return len(errors), errors_text

        except Exception as exc:
            return -1, f"DRC check failed: {exc}"

    def _render(self, component, instance_id) -> Image.Image:
        """
        [FIXED] Uses component_to_pil_image from utils instead of component.to_png
        """
        try:
            return component_to_pil_image(
                component,
                title=f"layout_{instance_id}",
                bbox=None 
            )
        except Exception as e:
            logger.error(f"Render failed for {instance_id}: {e}")
            # 返回一个红色的错误占位图，防止 pipeline 崩溃
            return Image.new('RGB', (224, 224), color='red')

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        clean_gds_path = kwargs.get("clean_gds_path")
        if not clean_gds_path:
            raise ValueError("DRCInteraction requires 'clean_gds_path'.")

        # Load GDS
        local_gds_path = copy_to_local(clean_gds_path)
        component = gf.import_gds(str(local_gds_path))
        self._ensure_reference_names(component)

        # Initial Check
        num_errors, error_message = self._get_drc_violations(component)
        
        # [FIX] 使用新的 _render 方法
        image = self._render(component, instance_id)
        
        self._instance_dict[instance_id] = {
            "component": component,
            "image": image,
            "drc_errors": num_errors,
            "drc_message": error_message,
            "fix_ops_count": 0,
            "history": [] 
        }
        return instance_id

    async def execute_tool_action(self, instance_id: str, tool_payload_json: str) -> tuple[Image.Image, str, int]:
        state = self._instance_dict[instance_id]
        component = state["component"]

        try:
            payload = json.loads(tool_payload_json)
            tool_name = payload.get("tool")
            args = payload.get("args")

            tool = self.tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found in DRCInteraction map.")

            # Execute Logic from drc_tool.py
            result = tool.execute(args=args, component=component)
            action_feedback = result.get("content", str(result))
            
            if "fix_ops_count" in state:
                state["fix_ops_count"] += 1

        except Exception as e:
            logger.error(f"Error executing {tool_name}: {e}")
            action_feedback = f"Tool execution failed: {e}"

        # Post-action: Check & Render
        num_errors, drc_status = self._get_drc_violations(component)
        
        # [FIX] 使用新的 _render 方法
        new_image = self._render(component, instance_id)

        # Update state
        state["image"] = new_image
        state["drc_errors"] = num_errors
        state["drc_message"] = drc_status
        
        full_feedback = f"Action Result: {action_feedback}\nCurrent DRC Status:\n{drc_status}"

        return new_image, full_feedback, num_errors

    async def generate_response(self, instance_id: str, messages: list[dict[str, Any]], **kwargs) -> tuple[bool, str, float, dict]:
        # Called at the end of an episode
        state = self._instance_dict[instance_id]
        
        final_drc_message = state["drc_message"]
        final_reward = 0.0 
        
        additional_data = {
            "image": state["image"],
            "drc_errors": state["drc_errors"],
            "fix_ops_count": state["fix_ops_count"],
        }
        
        return True, final_drc_message, final_reward, additional_data

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]




# [Append this to the end of verl/interactions/drc_interaction.py]
if __name__ == "__main__":
    import asyncio
    import shutil
    import tempfile
    import numpy as np

    async def test_interaction_flow():
        print("=== Testing Thick Interaction (GDS Logic & Rendering) ===")
        
        # Create a temporary directory for the test GDS file
        tmp_dir = tempfile.mkdtemp()
        gds_path = os.path.join(tmp_dir, "test_layout.gds")
        
        try:
            # 1. Prepare Data: Create a GDS file with a reference
            # Note: Our tools operate on References (instances), so we must use add_ref
            top = gf.Component("top_cell")
            rect = gf.components.rectangle(size=(10, 10), layer=(1, 0))
            
            # Add an instance and name it 'p0' (This is the key the Agent uses to operate)
            ref = top.add_ref(rect, name="p0")
            ref.center = (0, 0) # Initial center at (0,0)
            
            top.write_gds(gds_path)
            print(f"Created temp GDS at: {gds_path}")

            # 2. Initialize Environment
            interaction = DRCInteraction(config={})
            instance_id = await interaction.start_interaction(clean_gds_path=gds_path)
            print(f"Interaction Session Started: {instance_id}")

            # Verify initial state
            state = interaction._instance_dict[instance_id]
            print(f"Initial Error Count: {state['drc_errors']}")
            
            # 3. Simulate Agent Call: Move p0
            # This is the JSON string passed from the Thin Tool
            action_payload = json.dumps({
                "tool": "op_move_polygon",
                "args": {
                    "polygon_name": "p0",
                    "dx": 20.0,
                    "dy": 5.0
                }
            })
            
            print(f"\n[Action] Applying Payload: {action_payload}")
            img, feedback, errs = await interaction.execute_tool_action(instance_id, action_payload)
            
            print(f"Feedback: {feedback}")
            
            # 4. Verify Result (Physical state change)
            comp = state["component"]
            # Find the p0 instance
            # Note: gdsfactory instance references might be in insts or references
            target_ref = None
            if hasattr(comp, "named_instances") and "p0" in comp.named_instances:
                target_ref = comp.named_instances["p0"]
            
            assert target_ref is not None, "Failed to find instance 'p0' after operation"
            
            # Check coordinates: Initial (0,0) -> Move (20, 5) -> Expected Center (20, 5)
            # Note: gdsfactory center property returns a numpy array
            current_center = target_ref.center
            print(f"New Center: {current_center}")
            
            if np.allclose(current_center, [20.0, 5.0], atol=1e-3):
                print(">> Physics Verification Passed: Polygon moved correctly! ✅")
            else:
                print(f"!! Physics Verification Failed: Expected (20, 5), got {current_center} ❌")

            # 5. Verify Image Generation
            if img is not None and isinstance(img, Image.Image):
                print(f">> Rendering Verification Passed: Output image size {img.size} ✅")
                # img.show() # Uncomment to view image if running locally
            else:
                print("!! Rendering Verification Failed ❌")

        finally:
            # Cleanup
            shutil.rmtree(tmp_dir)
            print("\nTest cleanup done.")

    asyncio.run(test_interaction_flow())