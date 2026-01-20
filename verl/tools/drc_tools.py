# verl/tools/drc_tools.py

import logging
import os
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MovePolygonTool(BaseTool):
    """Tool to move a polygon in the layout."""
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "op_move_polygon",
                "description": "Moves a specified polygon by a given delta in x and y coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "polygon_id": {"type": "string", "description": "The unique identifier of the polygon to move."},
                        "dx": {"type": "number", "description": "The distance to move the polygon along the x-axis."},
                        "dy": {"type": "number", "description": "The distance to move the polygon along the y-axis."}
                    },
                    "required": ["polygon_id", "dx", "dy"],
                },
            }
        })
        super().__init__(config, _tool_schema)
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # This tool's execute method only returns a structured response.
        # The actual layout manipulation is handled by the DRCInteraction environment.
        polygon_id = parameters.get("polygon_id")
        dx = parameters.get("dx")
        dy = parameters.get("dy")
        
        # The text response signals the action to the interaction environment.
        # This is a convention for this specific implementation.
        response_text = f"[ACTION:MOVE_POLYGON] id={polygon_id} dx={dx} dy={dy}"
        return ToolResponse(text=response_text), 0.0, {}

class DeletePolygonTool(BaseTool):
    """Tool to delete a polygon from the layout."""
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "op_delete_polygon",
                "description": "Deletes a specified polygon from the layout.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "polygon_id": {"type": "string", "description": "The unique identifier of the polygon to delete."},
                    },
                    "required": ["polygon_id"],
                },
            }
        })
        super().__init__(config, _tool_schema)
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        polygon_id = parameters.get("polygon_id")
        response_text = f"[ACTION:DELETE_POLYGON] id={polygon_id}"
        return ToolResponse(text=response_text), 0.0, {}

class OffsetPolygonTool(BaseTool):
    """Tool to offset (resize) a polygon."""
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "op_offset_polygon",
                "description": "Resizes a polygon by a given offset. A positive offset expands it, a negative one shrinks it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "polygon_id": {"type": "string", "description": "The unique identifier of the polygon to offset."},
                        "offset": {"type": "number", "description": "The distance to offset the polygon's edges."},
                    },
                    "required": ["polygon_id", "offset"],
                },
            }
        })
        super().__init__(config, _tool_schema)
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        polygon_id = parameters.get("polygon_id")
        offset = parameters.get("offset")
        response_text = f"[ACTION:OFFSET_POLYGON] id={polygon_id} offset={offset}"
        return ToolResponse(text=response_text), 0.0, {}
        
class SplitPolygonTool(BaseTool):
    """Tool to split a polygon along a specified axis."""
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "op_split_polygon",
                "description": "Splits a polygon into two new polygons along a specified axis and position.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "polygon_id": {"type": "string", "description": "The unique identifier of the polygon to split."},
                        "axis": {"type": "string", "enum": ["x", "y"], "description": "The axis along which to split ('x' or 'y')."},
                        "position": {"type": "number", "description": "The coordinate on the axis where the split occurs."},
                    },
                    "required": ["polygon_id", "axis", "position"],
                },
            }
        })
        super().__init__(config, _tool_schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        polygon_id = parameters.get("polygon_id")
        axis = parameters.get("axis")
        position = parameters.get("position")
        response_text = f"[ACTION:SPLIT_POLYGON] id={polygon_id} axis={axis} position={position}"
        return ToolResponse(text=response_text), 0.0, {}

if __name__ == '__main__':
    # Simple test to verify schema creation
    print("Testing DRC Tool Schemas...")
    move_tool = MovePolygonTool(config={}, tool_schema=None)
    delete_tool = DeletePolygonTool(config={}, tool_schema=None)
    offset_tool = OffsetPolygonTool(config={}, tool_schema=None)
    split_tool = SplitPolygonTool(config={}, tool_schema=None)

    print("\nMove Tool Schema:")
    print(move_tool.get_openai_tool_schema().model_dump_json(indent=2))
    
    print("\nDelete Tool Schema:")
    print(delete_tool.get_openai_tool_schema().model_dump_json(indent=2))

    print("\nOffset Tool Schema:")
    print(offset_tool.get_openai_tool_schema().model_dump_json(indent=2))

    print("\nSplit Tool Schema:")
    print(split_tool.get_openai_tool_schema().model_dump_json(indent=2))
    print("\nDRC Tool schema tests passed!")

