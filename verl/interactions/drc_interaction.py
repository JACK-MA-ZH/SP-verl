# verl/interactions/drc_interaction.py

import logging
import os
import json
import asyncio
import tempfile
from typing import Any, Optional, Dict, List,Iterable
from uuid import uuid4
import traceback

import gdsfactory as gf
from PIL import Image

# [CRITICAL IMPORT] 从你的工具库中导入渲染函数
from verl.utils.drc.drc_tool import (
    MovePolygonTool, 
    # DeletePolygonTool, 
    # OffsetPolygonTool, 
    SplitPolygonTool,
    component_to_pil_image,  # <--- 必须使用这个函数来渲染
    _ensure_named_instance_map,
    _register_reference_name,
    _iter_references,
    _get_kdb_cell,
    _build_reference_snapshot,
    _format_netlist_payload
)
from gdsfactory.boolean import get_ref_shapes
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
            # "op_delete_polygon": DeletePolygonTool(),
            # "op_offset_polygon": OffsetPolygonTool(),
            "op_split_polygon": SplitPolygonTool(),
        }
    def get_schematic(self, component) -> str:
        if component is None:
            return "{}"
        netlist_payload = _format_netlist_payload(component)
        if netlist_payload:
            return netlist_payload
        snapshot = _build_reference_snapshot(component)
        return json.dumps(snapshot)
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

    # def _get_drc_violations(self, component) -> tuple[int, str]:
    #     """Runs DRC checks strictly using klayout.db."""
    #     import klayout.db as kdb

    #     if component is None:
    #         return 0, "No component loaded."

    #     def _bbox_to_tuple(box, dbu: float) -> tuple[float, float, float, float]:
    #         return (
    #             float(box.left) * dbu,
    #             float(box.bottom) * dbu,
    #             float(box.right) * dbu,
    #             float(box.top) * dbu,
    #         )

    #     # Target DRC rules from your second snippet
    #     min_spacing = 0.1
    #     min_width = 0.01

    #     try:
    #         layout = component.kcl.layout
    #         dbu = float(getattr(layout, "dbu", 1.0) or 1.0)
    #         layer_index = int(layout.layer(1, 0))

    #         if layer_index < 0:
    #             return 0, "No DRC errors found."

    #         region = kdb.Region()
            
    #         # 1. Fetch shapes from the main cell
    #         kdb_cell = _get_kdb_cell(component)
    #         if kdb_cell is not None:
    #             region += kdb.Region(kdb_cell.shapes(layer_index))
                
    #         # 2. Fetch shapes from references
    #         references = getattr(component, "named_instances", None)
    #         if isinstance(references, dict) and references:
    #             refs_iter = references.values()
    #         else:
    #             refs_iter = _iter_references(component)

    #         if get_ref_shapes is None:
    #             raise RuntimeError("get_ref_shapes helper unavailable; check drc_tool import")

    #         for ref in refs_iter:
    #             try:
    #                 region += get_ref_shapes(ref, layer_index)
    #             except Exception:
    #                 continue

    #         errors = []

    #         # 3. Perform spacing check
    #         spacing_pairs = list(region.space_check(min_spacing / dbu).each())
    #         for pair in spacing_pairs:
    #             bbox = _bbox_to_tuple(pair.bbox(), dbu)
    #             errors.append({"type": "min_spacing", "bbox": bbox})

    #         # 4. Perform width check
    #         width_pairs = list(region.width_check(min_width / dbu).each())
    #         for pair in width_pairs:
    #             bbox = _bbox_to_tuple(pair.bbox(), dbu)
    #             errors.append({"type": "min_width", "bbox": bbox})

    #         # 5. Format the output
    #         errors_text = (
    #             "\n".join([f"ERROR: {e['type']} at {e['bbox']}" for e in errors])
    #             if errors else "No DRC errors found."
    #         )
            
    #         return len(errors), errors_text

    #     except Exception as exc:
    #         return -1, f"DRC check failed: {exc}"
    def _get_drc_violations(self, component) -> tuple[int, str]:
        """Runs DRC checks strictly using klayout.db, preventing polygon merge."""
        import klayout.db as kdb

        if component is None:
            return 0, "No component loaded."

        def _bbox_to_tuple(box, dbu: float) -> tuple[float, float, float, float]:
            return (
                float(box.left) * dbu,
                float(box.bottom) * dbu,
                float(box.right) * dbu,
                float(box.top) * dbu,
            )

        # Target DRC rules
        min_spacing = 0.1
        min_width = 0.01

        try:
            layout = component.kcl.layout
            dbu = float(getattr(layout, "dbu", 1.0) or 1.0)
            layer_index = int(layout.layer(1, 0))

            if layer_index < 0:
                return 0, "No DRC errors found."

            min_spacing_dbu = int(min_spacing / dbu)
            min_width_dbu = int(min_width / dbu)

            # 核心改动：不再把所有东西塞进一个 Region，而是存成 Region 列表
            regions = []

            # 1. 获取主单元的图形（每个图形作为一个独立的 Region）
            kdb_cell = _get_kdb_cell(component)
            if kdb_cell is not None:
                for shape in kdb_cell.shapes(layer_index).each():
                    if shape.is_polygon() or shape.is_box() or shape.is_path():
                        regions.append(kdb.Region(shape.polygon))
                
            # 2. 获取子引用的图形（每个引用作为一个独立的 Region）
            references = getattr(component, "named_instances", None)
            if isinstance(references, dict) and references:
                refs_iter = references.values()
            else:
                refs_iter = _iter_references(component)

            if get_ref_shapes is None:
                raise RuntimeError("get_ref_shapes helper unavailable; check drc_tool import")

            for ref in refs_iter:
                try:
                    ref_region = get_ref_shapes(ref, layer_index)
                    if not ref_region.is_empty():
                        regions.append(ref_region)
                except Exception:
                    continue

            errors = []

            # 3. 对每个独立的 Region 进行内部检查 (线宽，以及子引用内部的间距)
            for r in regions:
                # 检查线宽
                width_pairs = list(r.width_check(min_width_dbu).each())
                for pair in width_pairs:
                    bbox = _bbox_to_tuple(pair.bbox(), dbu)
                    errors.append({"type": "min_width", "bbox": bbox})
                
                # 如果这个 Region 里有多个图形（比如一个子组件），检查它内部的间距
                space_pairs = list(r.space_check(min_spacing_dbu).each())
                for pair in space_pairs:
                    bbox = _bbox_to_tuple(pair.bbox(), dbu)
                    errors.append({"type": "min_spacing (internal)", "bbox": bbox})

            # 4. 跨 Region 检查（避免 Merge 问题的关键所在）
            num_regions = len(regions)
            for i in range(num_regions):
                r1 = regions[i]
                # BBox 快速排查，避免 O(N^2) 导致的卡顿
                bbox1_expanded = r1.bbox().enlarged(min_spacing_dbu, min_spacing_dbu)

                for j in range(i + 1, num_regions):
                    r2 = regions[j]

                    if not bbox1_expanded.overlaps(r2.bbox()):
                        continue  # 距离很远，跳过

                    # 4.1 检查重叠 (因为 separation_check 不抓重叠，所以必须补上这个)
                    overlap = r1 & r2
                    if not overlap.is_empty():
                        bbox = _bbox_to_tuple(overlap.bbox(), dbu)
                        errors.append({"type": "overlap", "bbox": bbox})
                        continue

                    # 4.2 检查间距 (跨 Region 的 space check)
                    sep_pairs = list(r1.separation_check(r2, min_spacing_dbu).each())
                    for pair in sep_pairs:
                        bbox = _bbox_to_tuple(pair.bbox(), dbu)
                        errors.append({"type": "min_spacing (inter-region)", "bbox": bbox})

            # 5. 格式化输出
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
        
        return component_to_pil_image(
                component,
                title=f"layout_{instance_id}",
                bbox=None 
            )
        # except Exception as e:
        #     logger.error(f"Render failed for {instance_id}: {e}")
        #     # 返回一个红色的错误占位图，防止 pipeline 崩溃
        #     return Image.new('RGB', (224, 224), color='red')

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        clean_gds_path = kwargs.get("clean_layout_gds_path")
        if not clean_gds_path:
            raise ValueError("DRCInteraction requires 'clean_layout_gds_path'.")

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
            "history": [] ,
            "move_penalty": 0
        }
        return instance_id

    async def execute_tool_action(self, instance_id: str, tool_payload_json: str) -> tuple[Image.Image, str, int]:
        state = self._instance_dict[instance_id]
        component = state["component"]

        move_penalty=0
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
            
            if tool_name=="op_move_polygon":
                dx = float(args["dx"])
                dy = float(args["dy"])
                move_penalty=abs(dx)+abs(dy)
                state["move_penalty"] += move_penalty
            if "fix_ops_count" in state:
                state["fix_ops_count"] += 1

        except Exception as e:
            #traceback.print_exc()
            logger.error(f"Error executing tool: {e}")#{tool_name}
            action_feedback = f"Tool execution failed: {e}"

        # Post-action: Check & Render
        num_errors, drc_status = self._get_drc_violations(component)
        
        # [FIX] 使用新的 _render 方法
        new_image = self._render(component, instance_id)

        # Update state
        state["image"] = new_image
        state["drc_errors"] = num_errors
        state["drc_message"] = drc_status
        component_schematic=self.get_schematic(component)
        full_feedback = f"Current Schematic: {component_schematic}\nAction Result: {action_feedback}\nCurrent DRC Status:\n{drc_status}"

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
            instance_id = await interaction.start_interaction(clean_layout_gds_path=gds_path)
            print(f"Interaction Session Started: {instance_id}")

            # Verify initial state
            state = interaction._instance_dict[instance_id]
            print(f"Initial Error Count: {state['drc_errors']}")
            
            # 3. Simulate Agent Call: Move p0
            # This is the JSON string passed from the Thin Tool
            
            action_payload = json.dumps({
                "tool": "op_split_polygon",
                "args": {
                    "polygon_name": "p0",
                    "split_line":{
                    "axis": "x",
                    "value": 0}
                }
            })
            
            print(f"\n[Action] Applying Payload: {action_payload}")
            img, feedback, errs = await interaction.execute_tool_action(instance_id, action_payload)
            
            print(f"Feedback: {feedback}")
            action_payload = json.dumps({
                "tool": "op_move_polygon",
                "args": {
                    "polygon_name": "p0_part1",
                    "dx": 2.0,
                    "dy": 0.0
                }
            })
            
            print(f"\n[Action] Applying Payload: {action_payload}")
            img, feedback, errs = await interaction.execute_tool_action(instance_id, action_payload)
            
            print(f"Feedback: {feedback}")
            
            action_payload = json.dumps({
                "tool": "op_split_polygon",
                "args": {
                    "polygon_name": "p0_part1",
                    "split_line":{
                    "axis": "y",
                    "value": 0}
                }
            })
            
            print(f"\n[Action] Applying Payload: {action_payload}")
            img, feedback, errs = await interaction.execute_tool_action(instance_id, action_payload)
            
            print(f"Feedback: {feedback}")
            action_payload = json.dumps({
                "tool": "op_split_polygon",
                "args": {
                    "polygon_name": "p0_part2",
                    "split_line":{
                    "axis": "x",
                    "value": 0}
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
            
            #assert target_ref is not None, "Failed to find instance 'p0' after operation"
            
            # Check coordinates: Initial (0,0) -> Move (20, 5) -> Expected Center (20, 5)
            # Note: gdsfactory center property returns a numpy array
            #current_center = target_ref.center
            #print(f"New Center: {current_center}")
            
            # if np.allclose(current_center, [20.0, 5.0], atol=1e-3):
            #     print(">> Physics Verification Passed: Polygon moved correctly! ✅")
            # else:
            #     print(f"!! Physics Verification Failed: Expected (20, 5), got {current_center} ❌")

            # 5. Verify Image Generation
            if img is not None and isinstance(img, Image.Image):
                print(f">> Rendering Verification Passed: Output image size {img.size} ✅")
                img.save('dog.png')
                #img.show() # Uncomment to view image if running locally
            else:
                print("!! Rendering Verification Failed ❌")

        finally:
            # Cleanup
            shutil.rmtree(tmp_dir)
            print("\nTest cleanup done.")

    asyncio.run(test_interaction_flow())