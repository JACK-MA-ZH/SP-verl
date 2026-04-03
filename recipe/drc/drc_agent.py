# verl/experimental/agent_loop/drc_agent_loop.py

import logging
import os
from typing import Any
from uuid import uuid4
import copy
import re
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop,AgentData
from verl.interactions.drc_interaction import DRCInteraction
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
import gdsfactory as gf
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
try:
    from verl.experimental.reward.reward_loop.registry import register as register_exp_rm
    
    @register_exp_rm("drc")
    class DummyExperimentalDRCReward:
        def __init__(self, config, *args, **kwargs):
            self.config = config

        def __call__(self, data, *args, **kwargs):
            return {"reward_score": 0.0, "reward_extra_info": {}}
except ImportError:
    pass
class DRCAgentData(AgentData):
    """Encapsulates all state variables for the DRC agent loop."""
    def __init__(self, messages, image_data, metrics, request_id, tools_kwargs, interaction, interaction_kwargs):
        super().__init__(messages, image_data, metrics, request_id, tools_kwargs, interaction, interaction_kwargs)
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

        self.phase = "gen"
        self.active_tool_schemas = []
        self.drc_errors_at_start = 0
        self.drc_errors_at_end = 0
        self.fix_ops_count = 0
        self.move_penalty=0
        self.format_score = 0.0
        self.last_format_score = 0.0
        #self.assistant_turns = 0

def get_format_reward(text, available_polygons):
            score = 0.0
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            
            if think_match:
                score += 2.5 # 基础格式分
                think_content = think_match.group(1)
                
                # 检查思考内容里是否提到了多边形的名字（证明它在观察环境）
                has_polygon = any(p in think_content for p in available_polygons)
                if has_polygon:
                    score += 2.5 # 提到了关键实体，额外加分！
                # else:
                #     score -= 0.5 # 胡言乱语没提到多边形，扣分
            else:
                score -= 5 # 没写标签，重罚
                
            return score

class DRCAgentLoop(ToolAgentLoop):
    """
    An agent loop specialized for DRC tasks. It orchestrates the multi-turn
    interaction between the LLM and the DRCInteraction environment.
    """
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        
        phase = kwargs.get("phase", "gen")
        
        strict_regex = kwargs.get("regex")
        #raise TypeError("drc going")
        # Initial prompt for the generator
        # [FIX] 1. 立即从 kwargs 提取 UID，确保全作用域可用
        # kwargs 是从 DataProto.non_tensor_batch 中解包出来的单个样本数据
        uid = kwargs.get("uid", f"unknown_{uuid4().hex}")
        
        sample_id = f"{uid}_{uuid4().hex[:8]}"
        
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        # Initial layout image and GDS path
        multi_modal_data = copy.deepcopy(kwargs["multi_modal_data"])
        # Initial "clean" layout image
        initial_image = copy.deepcopy(multi_modal_data["image"])
        
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        interaction_kwargs = kwargs.get("interaction_kwargs", {})
        
        #interaction_kwargs["clean_layout_gds_path"] = kwargs["clean_layout_gds_path"]
        
        # extra_info = kwargs.get("extra_info", {})
        # interaction_kwargs["clean_layout_gds_path"] = extra_info.get("clean_layout_gds_path")
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
        
        agent_data.phase = phase
        if phase == "gen":
            allowed_tools = ["op_split_polygon"]
        else:
            allowed_tools = ["op_move_polygon"]
        agent_data.active_tool_schemas = [
            t for t in self.tool_schemas if t["function"]["name"] in allowed_tools
        ]
        state = AgentState.PENDING
        turn_count = 1
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                # ==========================================
                # 1. 打印 LLM 的输入 (当前所有的历史消息)
                # ==========================================
                with open('llm_output_32b.log', 'a') as f:
                    f.write(f"\n\n{'='*20} [TURN {turn_count}] LLM INPUT {'='*20}")
                # 为了美观，可以只打印最后两三条，或者完整打印
                for msg in agent_data.messages:
                    # 如果 content 是列表(包含图片字典)，截断打印避免刷屏
                    content_str = str(msg['content'])
                    # if len(content_str) > 500:
                    #     content_str = content_str[:500] + " ... [TRUNCATED]"
                    with open('llm_output_32b.log', 'a') as f:
                        f.write(f"[{msg['role'].upper()}]: {content_str}")
                with open('llm_output_32b.log', 'a') as f:
                    f.write(f"{'='*60}\n")
                
                current_sampling_params = copy.deepcopy(sampling_params)
                # current_state = agent_data.interaction._instance_dict.get(request_id, {})
                # component = current_state.get("component")
                # available_polygons = [inst.name for inst in component.insts]
                # if available_polygons:
                #     poly_pattern = "|".join(map(re.escape, available_polygons))   
                # if agent_data.phase == "fix":
                #     current_sampling_params["regex"] = (
                #                 r"<think>[\s\S]*</think>\n?"
                #                 r"<tool_call>\n?"
                #                 r'\{"name": "op_move_polygon", "arguments": '
                #                 r'\{"polygon_name": "(' + poly_pattern + r')", '
                #                 r'"dx": -?[0-9]+(\.[0-9]+)?, "dy": -?[0-9]+(\.[0-9]+)?\}\}'
                #                 r'\n?</tool_call>'
                #             )
                # else:
                #     current_sampling_params["regex"] = (
                #                 r"<think>[\s\S]*</think>\n?"
                #                 r"<tool_call>\n?"
                #                 r'\{"name": "op_split_polygon", "arguments": '
                #                 r'\{"polygon_name": "(' + poly_pattern + r')", '
                #                 r'"axis": "[xy]", "value": -?[0-9]+(\.[0-9]+)?\}\}'
                #                 r'\n?</tool_call>'
                #             )
                    
                    
                state = await self._handle_generating_state(agent_data, current_sampling_params)#ignore_termination=True
                
                
                # ==========================================
                last_msg = agent_data.messages[-1]
                if last_msg["role"] == "assistant":
                    content = last_msg.get("content", "")
                    # 处理 content 可能是 list 的情况
                    text_content = "".join([c.get("text", "") for c in content if c.get("type") == "text"]) if isinstance(content, list) else str(content)
                    
                    # 1. 从当前交互环境中实时获取最新的 polygon 列表
                    current_state = agent_data.interaction._instance_dict.get(request_id, {})
                    component = current_state.get("component")
                    if component is not None:
                        available_polygons = [inst.name for inst in component.insts]
                    else:
                        available_polygons = [] # 兜底防错
                        
                    # 2. 调用外置函数进行打分
                    turn_format_score = get_format_reward(text_content, available_polygons)
                    agent_data.format_score += turn_format_score
                    agent_data.last_format_score = turn_format_score
                    # # 校验是否包含 <think> 标签
                    # if re.search(r"<think>.*?</think>", text_content, re.DOTALL):
                    #     agent_data.format_score += 1  # 乖乖思考了，加分 (W_FORMAT)
                    # else:
                    #     agent_data.format_score -= 0  # 直接动手不思考，扣分！(PENALTY_NO_THINK)

                # ==========================================
                # ==========================================
                # 2. 打印 LLM 的输出 (最新追加的 assistant 消息)
                # ==========================================
                with open('llm_output_32b.log', 'a') as f:
                    f.write(f"\n{'='*20} [TURN {turn_count}] LLM OUTPUT {'='*20}")
                    last_msg = agent_data.messages[-1]
                    f.write(f"[{last_msg['role'].upper()}]: {last_msg['content']}")
                    f.write(f"{'='*60}\n")
            elif state == AgentState.PROCESSING_TOOLS:
                prev_errors = agent_data.interaction._instance_dict[request_id].get("drc_errors", -1)
                state = await self._handle_drc_tool_processing(agent_data)
                current_errors = agent_data.interaction._instance_dict[request_id].get("drc_errors", -1)
                turn_count = turn_count+1
                # [新增] 完美修复，提前终止判定 (Early Stopping)
                # ==========================================
                if agent_data.phase == "gen":
                    if current_errors <= prev_errors:
                        # 错误没变少，甚至变多了！这是无效或负面动作，累计无效惩罚
                        agent_data.format_score -= agent_data.last_format_score
                        
                if agent_data.phase == "fix":
                    if current_errors >= prev_errors:
                        # 错误没变少，甚至变多了！这是无效或负面动作，累计无效惩罚
                        agent_data.format_score -= agent_data.last_format_score
                    # 从交互环境的字典中实时读取最新的 drc_errors
                   
                    if current_errors == 0:
                        logger.info(f"[DRCAgentLoop] UID: {uid} | Turn {turn_count}: DRC errors reached 0! Perfect fix. Terminating early.")
                        state = AgentState.TERMINATED  # 直接切断循环
                # ==========================================
            elif state == AgentState.INTERACTING:
                logger.info(f"[DRCAgentLoop] Model output text without tool call, terminating episode.")
                state = AgentState.TERMINATED
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
        
        # After the loop, get the final state from the interaction
        final_state = agent_data.interaction._instance_dict.get(request_id, {})
        agent_data.drc_errors_at_end = final_state.get("drc_errors", -1)
        agent_data.fix_ops_count = final_state.get("fix_ops_count", 0)
        
        save_dir = "/inspire/hdd/global_user/wuyouran-253108540218/llm/drc_generated_layouts_2"
        final_image = final_state.get("image")
        final_image.save(os.path.join(save_dir, f"{uid}_{sample_id}.png"))
        # 2. [关键修改] Loop 结束后，保存 GDS 状态到磁盘
        # 这样 Trainer 只需要知道 UID 就能找到对应的 GDS，不需要回传路径
        
        os.makedirs(save_dir, exist_ok=True)
        gds_save_path = os.path.join(save_dir, f"{uid}_{sample_id}.gds")
        # 调用 Interaction 的保存功能 (需要确保 Interaction 有这个接口，或者直接用 component write)
        final_state = agent_data.interaction._instance_dict.get(request_id, {})
        if "component" in final_state:
            final_state["component"].write_gds(gds_save_path)
        logger.info(f"[DRCAgentLoop] Saved GDS to {gds_save_path}")
        agent_data.drc_errors_at_end = final_state.get("drc_errors", -1)
        agent_data.fix_ops_count = final_state.get("fix_ops_count", 0)
        agent_data.move_penalty = final_state.get("move_penalty", 0)
 

        await interaction.release(request_id)
        
        # Finalize and prepare output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        last_drc_message = "No DRC errors found." # 默认兜底
        for msg in reversed(agent_data.messages):
            if msg["role"] == "tool":
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            last_drc_message = item.get("text")
                            break
                break
        drc_info = {
            "uid": uid,
            "drc_errors_before": agent_data.drc_errors_at_start,
            "drc_errors_after": agent_data.drc_errors_at_end,
            "num_fix_ops": agent_data.fix_ops_count,
            "final_drc_message": last_drc_message,
            "move_penalty": agent_data.move_penalty,
            "format_score": agent_data.format_score,
            "sample_id": sample_id,
            "phase": agent_data.phase
        }
        metrics_to_return={}
        
        metrics_to_return.update(agent_data.metrics)
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            multi_modal_data={"image": agent_data.image_data},
            response_logprobs=agent_data.response_logprobs[:self.response_length] if agent_data.response_logprobs else None,
            num_turns=len(agent_data.messages),
            metrics=metrics_to_return,
            extra_fields=drc_info
            
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
        
        new_messages = [
            {
                "role": "tool", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": drc_message}
                ]
            }
        ]
        
        
        # Update prompt with tool responses and the new image
        raw_tool_response_text = self.processor.apply_chat_template(
            new_messages,          
            #tools=agent_data.active_tool_schemas,#self.tool_schemas,      # 注意：必须再次带上工具 schema！, 
            add_generation_prompt=True, 
            tokenize=False, 
            **self.apply_chat_template_kwargs
        )
        model_inputs = self.processor(text=[raw_tool_response_text], images=[new_image], return_tensors="pt")
        response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        
        # Check length constraints
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        
        agent_data.messages.extend(new_messages)
        agent_data.image_data.append(new_image) # Add new image to the history
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