# verl/interactions/drc_interaction.py

import logging
import os
from typing import Any, Optional
from uuid import uuid4

import gdsfactory as gf
from PIL import Image

from verl.interactions.base import BaseInteraction
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class DRCInteraction(BaseInteraction):
    """
    Manages the state of a chip layout for a DRC task.
    This class simulates the environment by:
    - Loading an initial GDS layout.
    - Applying operations (move, delete, etc.) from tools.
    - Running a mock DRC check.
    - Rendering the layout to a PNG image.
    - Providing text feedback about the layout state.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    def _get_drc_errors_and_render(self, component: gf.Component) -> tuple[int, Image.Image, str]:
        """
        Runs a mock DRC check and renders the component to an image.
        In a real implementation, this would call a DRC tool like KLayout or Calibre.
        """
        # Mock DRC check: For this example, we'll just count polygons.
        # A more realistic mock would check for overlaps or spacing.
        num_errors = len(component.get_polygons()) % 5  # Mock error count
        
        # Render the component to a PNG image in memory
        png_buffer = component.to_png(resolution=300)
        image = Image.open(png_buffer).convert("RGB")

        # Mock DRC error message
        if num_errors > 0:
            error_message = f"DRC check found {num_errors} violations. Please review the layout."
        else:
            error_message = "DRC check passed successfully. No violations found."
            
        return num_errors, image, error_message

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Initializes a new DRC session with a clean layout."""
        if instance_id is None:
            instance_id = str(uuid4())

        clean_gds_path = kwargs.get("clean_gds_path")
        if not clean_gds_path:
            raise ValueError("DRCInteraction requires 'clean_gds_path' to start.")

        local_gds_path = copy_to_local(clean_gds_path)
        component = gf.import_gds(local_gds_path)

        num_errors, image, error_message = self._get_drc_errors_and_render(component)
        
        self._instance_dict[instance_id] = {
            "component": component,
            "image": image,
            "drc_errors": num_errors,
            "drc_message": error_message,
            "fix_ops_count": 0,
            "target_drc_rule": kwargs.get("target_drc_rule", "UNKNOWN"),
        }
        return instance_id

    async def generate_response(self, instance_id: str, messages: list[dict[str, Any]], **kwargs) -> tuple[bool, str, float, dict]:
        """
        Processes the agent's turn, which should contain a tool call action.
        This method is called by the `ToolAgentLoop` when it doesn't receive a tool call,
        so in our two-episode flow, this will be called at the end of each episode.
        
        For the DRC task, the "response" is the state of the layout after an action.
        """
        state = self._instance_dict[instance_id]
        
        # In our flow, `generate_response` is called when an agent *doesn't* call a tool,
        # which means an episode (gen or fix) has ended. We just return the final state.
        
        final_drc_message = state["drc_message"]
        final_reward = 0.0 # Reward is handled by the custom reward manager
        
        # Terminate the sequence, as the episode is over.
        should_terminate_sequence = True
        
        # The `multi_modal_outputs` will be packaged by the agent loop.
        # Here we just pass the necessary data.
        additional_data = {
            "image": state["image"],
            "drc_errors": state["drc_errors"],
            "fix_ops_count": state["fix_ops_count"],
        }
        
        return should_terminate_sequence, final_drc_message, final_reward, additional_data

    async def execute_tool_action(self, instance_id: str, action_text: str) -> tuple[Image.Image, str, int]:
        """
        This is a custom method called by the `DRCAgentLoop` to apply tool actions.
        It parses the action string from the tool and modifies the GDS component.
        """
        state = self._instance_dict[instance_id]
        component = state["component"]

        # Parse the action from the tool's text response
        # Example: "[ACTION:MOVE_POLYGON] id=poly1 dx=10 dy=0"
        action_parts = action_text.strip("[]").split()
        action_type = action_parts[0].split(":")[1]
        
        params = {}
        for part in action_parts[1:]:
            key, value = part.split("=")
            params[key] = value

        # Apply the action to the gdsfactory component
        # This is a mock implementation. A real one would need to identify
        # polygons by ID and apply precise transformations.
        try:
            if action_type == "MOVE_POLYGON":
                # In a real scenario, you'd select the polygon by `params['id']`
                # For this mock, we just move the first polygon found.
                if component.polygons:
                    component.polygons[0].move(origin=(0,0), destination=(float(params['dx']), float(params['dy'])))
            elif action_type == "DELETE_POLYGON":
                if component.polygons:
                    component.remove(component.polygons[0])
            elif action_type == "OFFSET_POLYGON":
                 if component.polygons:
                    component.polygons[0] = component.polygons[0].offset(float(params['offset']))
            elif action_type == "SPLIT_POLYGON":
                 if component.polygons:
                    # Mock split
                    poly = component.polygons[0]
                    component.remove(poly)
                    # This is highly simplified
                    component.add_polygon(poly.points[:len(poly.points)//2])
                    component.add_polygon(poly.points[len(poly.points)//2:])
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            # This is a fix operation
            if "fix_ops_count" in state:
                state["fix_ops_count"] += 1

        except Exception as e:
            logger.error(f"Error executing DRC tool action '{action_text}': {e}")

        # Re-run DRC and render the new layout
        num_errors, image, error_message = self._get_drc_errors_and_render(component)

        # Update state
        state["component"] = component
        state["image"] = image
        state["drc_errors"] = num_errors
        state["drc_message"] = error_message
        
        return image, error_message, num_errors

    async def release(self, instance_id: str, **kwargs) -> None:
        """Cleans up the state for a given instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

if __name__ == '__main__':
    # A simple test to ensure the interaction can be initialized and methods called.
    async def test_drc_interaction():
        print("Testing DRCInteraction...")
        config = {"name": "drc_interaction"}
        interaction = DRCInteraction(config)

        # Create a dummy GDS file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            c = gf.Component("test_component")
            c.add_polygon([(0,0), (10,0), (10,10), (0,10)])
            c.add_polygon([(20,0), (30,0), (30,10), (20,10)])
            gds_path = os.path.join(tmpdir, "test.gds")
            c.write_gds(gds_path)
            
            # Start interaction
            instance_id = await interaction.start_interaction(clean_gds_path=gds_path)
            print(f"Interaction started with instance ID: {instance_id}")
            initial_state = interaction._instance_dict[instance_id]
            print(f"Initial DRC errors: {initial_state['drc_errors']}")
            
            # Simulate a tool action
            action = "[ACTION:MOVE_POLYGON] id=poly1 dx=5 dy=0"
            new_image, new_message, new_errors = await interaction.execute_tool_action(instance_id, action)
            print(f"After action '{action}':")
            print(f"  New DRC message: {new_message}")
            print(f"  New DRC errors: {new_errors}")
            print(f"  Fix ops count: {interaction._instance_dict[instance_id]['fix_ops_count']}")
            assert new_image is not None
            
            # Simulate end of episode
            should_terminate, final_msg, _, _ = await interaction.generate_response(instance_id, messages=[])
            print(f"Final response: terminate={should_terminate}, message='{final_msg}'")
            
            # Release instance
            await interaction.release(instance_id)
            assert instance_id not in interaction._instance_dict
            print("Instance released successfully.")
            
        print("\nDRCInteraction test passed!")

    asyncio.run(test_drc_interaction())

