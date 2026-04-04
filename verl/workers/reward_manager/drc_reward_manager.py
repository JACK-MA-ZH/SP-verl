# verl/workers/reward_manager/drc_reward_manager.py

import logging
import os
from collections import deque
from typing import Any
from tensordict import TensorDict
import numpy as np
import torch
import re
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.registry import register

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@register("drc")
class DRCRewardManager(AbstractRewardManager):
    """
    A reward manager for the dual-agent DRC task. It calculates rewards for
    both the generator (π_gen) and the fixer (π_fix) agents based on the
    outcome of two back-to-back episodes. It also dynamically adjusts weights.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        
        # Constants for reward calculation
        self.C1_TARGET_HIT_POSITIVE = 5.0
        self.C1_TARGET_HIT_NEGATIVE = -5.0
        self.C2_DRC_CLEAN_PERFECT = 10.0
        self.WC_CHALLENGE_WEIGHT = 0.2
        self.WR_REDUCTION_WEIGHT = 5.0
        self.WP_PERTURBATION_PENALTY = 0.2
        self.W_FORMAT = 3 
        self.PENALTY_NO_THINK = -3.0
        # State for dynamic weighting
        self.fix_success_history = deque(maxlen=100)
        self.w_gen = 0.1
        self.w_fix = 0.1
        
        self.print_count = 0
        
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Calculates rewards for a combined batch of generator and fixer trajectories.
        The input `data` is expected to be a concatenation of trajectories from
        both agents for a set of initial layouts.
        """
        # The batch is interleaved: [gen_0, fix_0, gen_1, fix_1, ...]
        batch_size = data.batch.batch_size[0]
        #assert batch_size % 2 == 0, "Batch size must be even for gen/fix pairs."
        
        track_vars = {
            "gen_n_after": [],"n_before": [], "n_after": [], "n_fix_ops": [], 
            "r_gen": [], "r_fix": [], "final_r_gen": [], "final_r_fix": []
        }
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        half_batch = batch_size // 2
        phases = data.non_tensor_batch.get("phase", ["gen"] * batch_size)
        
        for i in range(0, batch_size):
            phase = phases[i]
            # Extract trajectories for one episode
            if phase == "gen":
                gen_traj = data[i]
                gen_format_score = gen_traj.non_tensor_batch.get("format_score", 0.0)
                gen_n_after=gen_traj.non_tensor_batch.get("drc_errors_after", 0)
                # --- Calculate R_gen ---
                r_target_hit = self.C1_TARGET_HIT_POSITIVE*gen_n_after #if n_before > 0 else self.C1_TARGET_HIT_NEGATIVE
            # r_challenge = self.WC_CHALLENGE_WEIGHT * n_fix_ops
                r_gen = r_target_hit + gen_format_score#+ r_challenge
                # --- Apply dynamic weights ---
                final_r_gen = self.w_gen * r_gen
                # --- Assign sparse rewards to the reward tensor ---
                gen_response_mask = gen_traj.batch["attention_mask"][gen_traj.batch["prompts"].shape[-1]:]
                gen_valid_len = int(gen_response_mask.sum().item())
                if gen_valid_len > 0:
                    reward_tensor[i, gen_valid_len - 1] = final_r_gen
                track_vars["gen_n_after"].append(gen_n_after)
                track_vars["r_gen"].append(r_gen)
                track_vars["final_r_gen"].append(final_r_gen)
            
            if phase == "fix":
                fix_traj = data[i]
                fix_format_score = fix_traj.non_tensor_batch.get("format_score", 0.0)
                n_before = fix_traj.non_tensor_batch.get("drc_errors_before", 0)
                n_after = fix_traj.non_tensor_batch.get("drc_errors_after", 0)
                n_fix_ops = fix_traj.non_tensor_batch.get("num_fix_ops", 0)
                move_penalty = fix_traj.non_tensor_batch.get("move_penalty", 0)
                # --- Calculate R_fix ---
                # if n_after == 0 and n_before!=0:
                #     r_drc_clean = self.C2_DRC_CLEAN_PERFECT
                # else:
                initial_errors_for_fix = fix_traj.non_tensor_batch.get("drc_errors_before", n_before)
                r_drc_clean = self.WR_REDUCTION_WEIGHT * (initial_errors_for_fix - n_after)
                move_penalty=min(200,move_penalty)
                r_perturbation = -self.WP_PERTURBATION_PENALTY * (n_fix_ops+move_penalty)
                r_fix = r_drc_clean + r_perturbation + fix_format_score
                final_r_fix = self.w_fix * r_fix
                # --- Update success history and adjust weights ---
                is_perfect_fix = (n_after == 0) and (n_before>0)
                self.fix_success_history.append(is_perfect_fix)
                
                fix_response_mask = fix_traj.batch["attention_mask"][fix_traj.batch["prompts"].shape[-1]:]
                
                fix_valid_len = int(fix_response_mask.sum().item())
            
                if fix_valid_len > 0:
                    reward_tensor[i, fix_valid_len - 1] = final_r_fix

            
                track_vars["n_before"].append(n_before)
                track_vars["n_after"].append(n_after)
                track_vars["n_fix_ops"].append(n_fix_ops)
                track_vars["r_fix"].append(r_fix)
                track_vars["final_r_fix"].append(final_r_fix)
            
            # Logging for debugging
            # if 1:#self.print_count < self.num_examine:
            #     print("-" * 20)
            #     print(f"Episode Pair {i}:")
            #     print(f"  N_before={n_before}, N_after={n_after}, N_fix_ops={n_fix_ops}")
            #     print(f"  R_gen = {r_gen:.2f}, R_fix = {r_fix:.2f}")
            #     print(f"  Dynamic Weights: w_gen={self.w_gen:.2f}, w_fix={self.w_fix:.2f}")
            #     print(f"  Final Rewards: R_gen={final_r_gen:.2f}, R_fix={final_r_fix:.2f}")
                # self.print_count += 1
        #self._adjust_dynamic_weights()        
        drc_metrics = {}
        if track_vars["n_before"]: # Ensure it's not empty
            for key, values in track_vars.items():
                drc_metrics[f"drc_env/{key}/mean"] = sum(values) / len(values)
                drc_metrics[f"drc_env/{key}/max"] = max(values)
                drc_metrics[f"drc_env/{key}/min"] = min(values)
        
        # Add the dynamic weights (these are scalars, so no min/max needed)
        drc_metrics["drc_env/w_gen"] = self.w_gen
        drc_metrics["drc_env/w_fix"] = self.w_fix        
        return {"reward_tensor": reward_tensor, "drc_metrics": drc_metrics} if return_dict else reward_tensor

    def _adjust_dynamic_weights(self):
        # if len(self.fix_success_history) < 50: # Wait for enough history
        #     return
            
        fix_success_rate = sum(self.fix_success_history) / len(self.fix_success_history)
        
        if fix_success_rate > 0.7: # Task is too easy
            self.w_gen = min(0.1, self.w_gen + 0.2)
            self.w_fix = max(0.0, self.w_fix - 0.2)
        elif fix_success_rate < 0.30: # Task is too hard
            self.w_gen = max(0.0, self.w_gen - 0.2)
            self.w_fix = min(0.1, self.w_fix + 0.2)

if __name__ == "__main__":
    # Simple test case for the reward manager
    print("Testing DRCRewardManager...")
    manager = DRCRewardManager(tokenizer=None, num_examine=5)

    # Mock DataProto for one pair of episodes
    mock_batch = {
        "prompts": torch.zeros(2, 10, dtype=torch.long),
        "responses": torch.zeros(2, 20, dtype=torch.long),
        "attention_mask": torch.cat([torch.ones(2, 10), torch.ones(2, 20)], dim=1),
    }
    mock_non_tensor = {
        "drc_errors_after": np.array([5, 0]), 
        "num_fix_ops": np.array([0, 12]), 
        "drc_errors_before": np.array([5, 0]),
    }
    
    data = DataProto(
        batch=TensorDict(mock_batch, batch_size=[2]),
        non_tensor_batch=mock_non_tensor
    )
    
    # Simulate a few steps
    print("\n--- Test 1: Fixer Succeeds ---")
    data.non_tensor_batch["drc_errors_after"][1] = 0 
    reward_dict = manager(data, return_dict=True)
    
    # Indices must be correct now
    print("Gen Reward:", reward_dict["reward_tensor"][0, 19].item())
    print("Fix Reward:", reward_dict["reward_tensor"][1, 19].item())
    
    print("\n--- Test 2: Fixer Fails ---")
    data.non_tensor_batch["drc_errors_after"][1] = 2 
    reward_dict = manager(data, return_dict=True)
    print("Gen Reward:", reward_dict["reward_tensor"][0, 19].item())
    print("Fix Reward:", reward_dict["reward_tensor"][1, 19].item())

    print("\nDRCRewardManager test completed.")


