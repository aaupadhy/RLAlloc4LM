
# RLAlloc4LM: Resource Allocation for LLMs using Deep Reinforcement Learning

**RLAlloc4LM** is a novel approach to dynamic resource allocation for Large Language Model training using Soft Actor-Critic with expert policy imitation. Building upon the work of Guo et al. in *"Learning-based Resource Management for Cloud Networks"*, we extend their framework specifically for LLM training workloads.

From Guo et al.:
> "Deep reinforcement learning demonstrates significant potential in cloud resource scheduling, achieving notable improvements over traditional heuristics by learning adaptive policies through experience."

## Key Features
- **Dynamic resource allocation** for GPU, CPU, and memory
- **Expert policy imitation** using SJF, FCFS, and Round Robin
- **Prioritized experience replay** with importance sampling
- **Continuous action space** for fine-grained control
- **Profiling integration** for GPT-style workloads

## Installation
```bash
git clone https://github.com/aaupadhy/RLAlloc4LM.git
cd RLAlloc4LM
conda create -n RLAlloc4LM python=3.9
conda activate RLAlloc4LM
pip install -r requirements.txt
```

## Project Structure
```
.
├── data/
│   └── logs/          # Training logs and traces
├── env/               # Environment implementation
├── experiments/       # Configuration files
├── results/           # Training results and plots
├── rlalloc/           # Core implementation
│   ├── agents/        # RL agents (SAC)
│   ├── experts/       # Expert policies
│   ├── models/        # Neural network architectures
│   └── utils/         # Utilities and helpers
├── scripts/           # Training and evaluation scripts
└── tests/             # Unit tests
```

## Quickstart
**Generate training traces:**
```bash
python scripts/profile_gpt.py
```

**Train the model:**
```bash
python scripts/train_rlalloc.py
```

**Evaluate and plot results:**
```bash
python scripts/evaluate_models.py
python scripts/plot_results.py
```

## Configuration
Key configuration in `experiments/configs/config.yaml`:
```yaml
environment:
  max_gpu_memory: 42949672960  # 40GB
  max_cpu_memory: 274877906944  # 256GB
  max_cpu_cores: 32
  time_horizon: 39
  max_steps: 1000

training:
  total_episodes: 200
  learning_rate: 0.003
  batch_size: 64
  buffer_size: 1000000
  gamma: 0.99
```

## Customization
**Adding New Expert Policies:**
```python
# In rlalloc/experts/custom_policy.py
class CustomPolicy:
    def get_demonstration(self, state):
        # Implement custom allocation logic
        return action_vector
```

**Modifying Reward Function:**
```python
# In env/resource_env.py
def _calculate_reward(self):
    # Implement custom reward logic
    return reward
```

**Adding Custom Metrics:**
```python
# In rlalloc/utils/metrics.py
def custom_metric(self):
    # Implement metric calculation
    return metric_value
```

## Running Tests
```bash
python -m pytest tests/
```

## License
MIT License - see LICENSE file for details.

## Citation
```bibtex
@ARTICLE{9200455,
  author={Guo, Wenxia and Tian, Wenhong and Ye, Yufei and Xu, Lingxiao and Wu, Kui},
  journal={IEEE Internet of Things Journal}, 
  title={Cloud Resource Scheduling With Deep Reinforcement Learning and Imitation Learning}, 
  year={2021},
  volume={8},
  number={5},
  pages={3576-3586},
  keywords={Resource management;Cloud computing;Machine learning;Task analysis;Dynamic scheduling;Processor scheduling;Learning (artificial intelligence);Cloud resource scheduling;deep reinforcement learning (deep RL);imitation learning},
  doi={10.1109/JIOT.2020.3025015}}
```

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Create Pull Request

## Contact
For questions or support, please open an issue on GitHub.

## Acknowledgments
We thank the authors of the original DeepRM paper and Guo et al. for their foundational work in applying deep reinforcement learning to resource management.
