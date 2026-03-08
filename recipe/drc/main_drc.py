# recipe/drc/main.py
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp

# 检查文件是否存在并删除
file_path = "llm_output.log"
if os.path.isfile(file_path):
    os.remove(file_path)
    print("文件已成功删除")
else:
    print("文件不存在")
mp.set_sharing_strategy('file_system')
# 1. 将项目根目录加入路径，确保能找到 recipe 模块
sys.path.append(os.getcwd())
sys.path.append(".")
# 2. 导入 verl 的核心组件
from verl.trainer.main_ppo import run_ppo
from verl.utils.reward_score import gsm8k, math_reward

# 3. [关键] 显式导入你的自定义代码，触发 @register 装饰器
# 确保这里引用的文件名与你实际的文件名一致
import recipe.drc.drc_agent
import verl.workers.reward_manager.drc_reward_manager
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

# 1. 定义一个配置路径初始化函数
def register_verl_configs():
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.config_loader import ConfigLoader
    
    # 获取 verl 库中配置文件的绝对路径
    import verl
    verl_config_path = os.path.join(os.path.dirname(verl.__file__), 'trainer/config')
    
    # 打印出来确认一下（调试用）
    print(f"Adding to search path: {verl_config_path}")
    return verl_config_path

# 2. 修改 main 函数及其装饰器
# 注意：我们将 config_path 指向你当前项目的 config 目录
@hydra.main(config_path="config", config_name='drc_trainer', version_base=None)
def main(cfg: DictConfig):
    run_ppo(cfg)

if __name__ == '__main__':
    # 在运行 main 之前，通过命令行参数动态注入搜索路径
    # 这样你就不需要复制 ppo_trainer.yaml 了
    import verl
    import os
    verl_path = os.path.join(os.path.dirname(verl.__file__), 'trainer/config')
    
    # 将 verl 的路径加入到 hydra 的搜索路径中
    sys.argv.append(f'--config-dir={verl_path}') 
    
    main()