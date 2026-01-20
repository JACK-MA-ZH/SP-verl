# verl/experimental/agent_loop/drc_agent_loop.py

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.interactions.drc_interaction import DRCInteraction
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class DRCAgentData:
    """Encapsulates all state variables for the DRC agent loop."""
    def __init__(self, messages, image_data, metrics, request_id, tools_kwargs, interaction, interaction_kwargs):
        self.messages = messages
        self.image_data = image_data # This will be a list of images over turns
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction: DRCInteraction = interaction
        self.interaction_kwargs = interaction_kwargs

        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []

        self.drc_errors_at_start = 0
        self.drc_errors_at_end = 0
        self.fix_ops_count = 0


@register("drc_agent")
class DRCAgentLoop(ToolAgentLoop):
    """
    An agent loop specialized for DRC tasks. It orchestrates the multi-turn
    interaction between the LLM and the DRCInteraction environment.
    """
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Initial prompt for the generator
        messages = list(kwargs["raw_prompt"])
        # Initial layout image and GDS path
        multi_modal_data = kwargs["multi_modal_data"]
        # Initial "clean" layout image
        initial_image = multi_modal_data["image"]
        
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        interaction_kwargs = kwargs.get("interaction_kwargs", {})
        
        # Initialize the DRC interaction environment
        interaction: DRCInteraction = self.interaction_map["drc_interaction"]
        await interaction.start_interaction(request_id, **multi_modal_data, **interaction_kwargs)
        
        agent_data = DRCAgentData(
            messages=messages,
            image_data=initial_image,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs
        )
        agent_data.drc_errors_at_start = interaction._instance_dict[request_id]["drc_errors"]
        
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params, ignore_termination=True)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_drc_tool_processing(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
        
        # After the loop, get the final state from the interaction
        final_state = agent_data.interaction._instance_dict.get(request_id, {})
        agent_data.drc_errors_at_end = final_state.get("drc_errors", -1)
        agent_data.fix_ops_count = final_state.get("fix_ops_count", 0)

        await interaction.release(request_id)
        
        # Finalize and prepare output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            multi_modal_data={"image": agent_data.image_data},
            response_logprobs=agent_data.response_logprobs[:self.response_length] if agent_data.response_logprobs else None,
            num_turns=len(agent_data.messages),
            metrics=agent_data.metrics,
            extra_fields={
                "drc_errors_before": agent_data.drc_errors_at_start,
                "drc_errors_after": agent_data.drc_errors_at_end,
                "num_fix_ops": agent_data.fix_ops_count,
            }
        )
        return output

    async def _handle_drc_tool_processing(self, agent_data: DRCAgentData) -> AgentState:
        """
        Custom tool processing for DRC. Instead of just adding a text response,
        this will call the interaction environment to update the layout state.
        """
        tool_call = agent_data.tool_calls[0] # Assuming one tool call per turn for simplicity
        
        with simple_timer("tool_calls", agent_data.metrics):
            # The tool `execute` just returns a formatted string.
            tool_response, _, _ = await self._call_tool(tool_call, agent_data.tools_kwargs)
        
        # The interaction environment executes the action.
        new_image, drc_message, _ = await agent_data.interaction.execute_tool_action(
            agent_data.request_id, tool_response.text
        )

        # The new layout state becomes the next "user" turn.
        # We pass the image and text feedback.
        # NOTE: This deviates from the standard `ToolAgentLoop` by adding a new image.
        
        new_messages = [{"role": "tool", "content": drc_message}]
        agent_data.messages.extend(new_messages)
        agent_data.image_data.append(new_image) # Add new image to the history
        
        # Update prompt with tool responses and the new image
        raw_tool_response_text = self.processor.apply_chat_template(
            new_messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
        )
        model_inputs = self.processor(text=[raw_tool_response_text], images=[new_image], return_tensors="pt")
        response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        
        # Check length constraints
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        return AgentState.GENERATING
        
if __name__ == "__main__":
    # This requires a more complex setup to test, involving mock tools and interactions.
    # A unit test here would be non-trivial.
    print("DRCAgentLoop defined. Requires integration test to validate.")

