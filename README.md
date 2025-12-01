# LeMix
Codebase for the RTSS 2025 paper: "LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems" ([PDF](https://arxiv.org/pdf/2507.21276)).


# Quick Start
To get started with LeMix, follow these steps:
1. **Clone the Repository**
    ```bash
    git clone https://github.com/yul091/LM-profiling.git
    cd LM-profiling
    ```
2. **Install Dependencies**
    Make sure you have Python 3.7+ installed. Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Example**
    You can run a sample script to see LeMix in action:
    ```bash
    bash sec_2.2.sh
    ```

# Reference
If you find LeMix useful in your research or projects, please consider citing the following paper:

```bibtex
@inproceedings{li2025lemix,
  title={LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems},
  author={Li, Yufei and Li, Zexin and Zhu, Yinglun and Liu, Cong},
  booktitle={{IEEE} Real-Time Systems Symposium (RTSS)},
  year={2025}
}
```