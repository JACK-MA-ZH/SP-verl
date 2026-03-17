# verl/tools/drc_tools.py

import logging
import os
import json
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MovePolygonTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # Schema matching your drc_tool.py MovePolygonTool
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "op_move_polygon",
                "description": "Moves a specified polygon (instance) by a given delta (dx, dy).",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "polygon_name": {"type": "string", "description": "Name of the polygon/instance to move."},
                        "dx": {"type": "number", "description": "Distance to move in x (um)."},
                        "dy": {"type": "number", "description": "Distance to move in y (um)."},
                    },
                    "required": ["polygon_name", "dx", "dy"],
                },
            }
        })
        super().__init__(config, _tool_schema)
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # Thin execution: Just serialize the call for the Interaction layer
        return ToolResponse(text=json.dumps({"tool": "op_move_polygon", "args": parameters})), 0.0, {}

# class DeletePolygonTool(BaseTool):
#     def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
#         _tool_schema = OpenAIFunctionToolSchema.model_validate({
#             "type": "function",
#             "function": {
#                 "name": "op_delete_polygon",
#                 "description": "Deletes a specified polygon/instance from the layout.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "polygon_name": {"type": "string", "description": "Name of the polygon/instance to delete."},
#                     },
#                     "required": ["polygon_name"],
#                 },
#             }
#         })
#         super().__init__(config, _tool_schema)
        
#     async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
#         return ToolResponse(text=json.dumps({"tool": "op_delete_polygon", "args": parameters})), 0.0, {}

# class OffsetPolygonTool(BaseTool):
#     def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
#         _tool_schema = OpenAIFunctionToolSchema.model_validate({
#             "type": "function",
#             "function": {
#                 "name": "op_offset_polygon",
#                 "description": "Offsets (shrinks/grows) a polygon by a specific distance on a specific layer.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "polygon_name": {"type": "string", "description": "Name of the polygon/instance to offset."},
#                         "distance": {"type": "number", "description": "Offset distance (um). Negative shrinks, positive grows."},
#                         "layer": {
#                             "type": "array",
#                             "items": {"type": "number"},
#                             "minItems": 2, "maxItems": 2,
#                             "description": "GDS Layer [layer, purpose], e.g., [1, 0]."
#                         },
#                     },
#                     "required": ["polygon_name", "distance", "layer"],
#                 },
#             }
#         })
#         super().__init__(config, _tool_schema)
        
    # async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
    #     return ToolResponse(text=json.dumps({"tool": "op_offset_polygon", "args": parameters})), 0.0, {}
        
class SplitPolygonTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "op_split_polygon",
                "description": "use an infinite straight line (x=value or y=value) to split a polygon.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "polygon_name": {"type": "string", "description": "Name of the polygon/instance to split."},
                        
                        "axis": {
                                    "type": "string",
                                    "enum": ["x", "y"],
                                    "description": "decide the line is x=value or y=value?",
                                },
                        "value": {"type": "number", "description": "line's value"},
                            },
                    "required": ["polygon_name", "axis","value"],
                },
            }
        })
        super().__init__(config, _tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text=json.dumps({"tool": "op_split_polygon", "args": parameters})), 0.0, {}
# [Append this to the end of verl/tools/drc_tools.py]
if __name__ == "__main__":
    import asyncio
    import json

    async def test_tools():
        print("=== Testing Thin Tools (Schema & Protocol) ===")
        
        # 1. 初始化工具
        # 注意: BaseTool 通常需要 config 和 tool_schema，这里模拟传入
        # 如果你的 BaseTool 不需要 config，可以传空字典
        move_tool = MovePolygonTool(config={}, tool_schema=None)
        split_tool = SplitPolygonTool(config={}, tool_schema=None)

        # 2. 验证 Schema 生成 (Agent 看到的说明书)
        print("\n[MoveTool Schema]:")
        schema = move_tool.tool_schema.model_dump()
        print(json.dumps(schema["function"]["parameters"], indent=2))
        
        # 3. 验证执行逻辑 (应该只返回 JSON 字符串)
        print("\n[Executing MoveTool]:")
        params = {"polygon_name": "p0", "dx": 10.5, "dy": -2.0}
        # execute 通常需要 instance_id，这里给个假的
        res, _, _ = await move_tool.execute(instance_id="test_run", parameters=params)
        
        print(f"Tool Output: {res.text}")
        
        # 校验输出是否符合 Interaction 层的协议
        parsed = json.loads(res.text)
        assert parsed["tool"] == "op_move_polygon"
        assert parsed["args"]["dx"] == 10.5
        print(">> MoveTool Verification Passed ✅")

        print("\n[Executing SplitTool]:")
        split_params = {
            "polygon_name": "p0", 
            "split_line_bbox": [0, 0, 10, 10], 
            "layer": [1, 0]
        }
        res_split, _, _ = await split_tool.execute(instance_id="test_run", parameters=split_params)
        print(f"Tool Output: {res_split.text}")
        
        parsed_split = json.loads(res_split.text)
        assert parsed_split["tool"] == "op_split_polygon"
        print(">> SplitTool Verification Passed ✅")

    asyncio.run(test_tools())