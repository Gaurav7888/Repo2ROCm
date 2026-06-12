# AMD/ROCm: out-of-box vs needs-repo2rocm (60 repos)

_OUT_OF_BOX = device-agnostic PyTorch that runs by following the README with at most ordinary pip installs. REQUIRES_REPO2ROCM = CUDA-only deps / custom CUDA kernels / cuda-pinned torch / nccl-distributed / syntax or import bugs / risky version pins / missing source._

- **OUT_OF_BOX:** 18
- **REQUIRES_REPO2ROCM:** 42

| Repo | SHA | Verdict | Conf | Blockers |
|---|---|---|---|---|
| `allenai/understanding_mcqa` | 8b1aea4c | OUT_OF_BOX | high | none |
| `BruceGeLi/TOP_ERL_ICLR25` | HEAD | OUT_OF_BOX | high | none |
| `dayeonki/askqe` | HEAD | OUT_OF_BOX | high | none |
| `liukidar/pcx` | HEAD | OUT_OF_BOX | high | none |
| `llmsresearch/paperbanana` | HEAD | OUT_OF_BOX | high | none |
| `microsoft/SeqSNN` | HEAD | OUT_OF_BOX | high | none |
| `mmoharami/Risk-Sensitive-CMDP` | HEAD | OUT_OF_BOX | high | none |
| `murtylab/topoloss` | HEAD | OUT_OF_BOX | high | none |
| `OpenBMB/IoA` | HEAD | OUT_OF_BOX | high | none |
| `parameterlab/c-seo-bench` | HEAD | OUT_OF_BOX | high | none |
| `pouyahmdn/LCPO` | HEAD | OUT_OF_BOX | high | and can be resolved by updating to compatible versions. |
| `qiuzh20/gated_attention` | HEAD | OUT_OF_BOX | high | by examining the requirements mentioned in the README: |
| `ryienh/jl-metric` | ae0a1e4c | OUT_OF_BOX | high | none |
| `sapienzanlp/bookcoref` | HEAD | OUT_OF_BOX | high | none |
| `sarahmart/HARDMath` | 9ade062a | OUT_OF_BOX | high | none |
| `SCUNLP/ELABORATION` | HEAD | OUT_OF_BOX | high | none |
| `shimo-lab/modelmap` | ce0826b8 | OUT_OF_BOX | high | none |
| `THUKElab/CLEME` | HEAD | OUT_OF_BOX | high | none |
| `1202kbs/GCTM` | HEAD | REQUIRES_REPO2ROCM | high | are the CUDA-pinned PyTorch versions in the README environment specification and the Pytho |
| `alessiopittiglio/mm-argfallacy` | HEAD | REQUIRES_REPO2ROCM | high | flash_attn==2.7.4.post1, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu1 |
| `allenai/WildBench` | HEAD | REQUIRES_REPO2ROCM | high | vllm dependency, nvidia-smi hardware detection, auto_gptq import, BitsAndBytesConfig usage |
| `apple/ml-tarflow` | HEAD | REQUIRES_REPO2ROCM | high | hardcoded NCCL backend in distributed training |
| `backprop07/Self-Certainty` | HEAD | REQUIRES_REPO2ROCM | high | hardcoded CUDA autocast, missing eval_utils module, missing templates module |
| `batmanlab/Ladder` | HEAD | REQUIRES_REPO2ROCM | high | CUDA-pinned PyTorch wheels, torch.backends.cudnn without HIP guards, missing source module |
| `codezakh/DataEnvGym` | HEAD | REQUIRES_REPO2ROCM | high | that would prevent it from running on AMD ROCm out of the box: |
| `Cohere-Labs-Community/m-rewardbench` | 708d3bde | REQUIRES_REPO2ROCM | high | vllm (needs rocm/vllm base image; pip vllm is CUDA-only) |
| `DCDmllm/HealthGPT` | HEAD | REQUIRES_REPO2ROCM | high | bitsandbytes==0.41.0, deepspeed==0.9.5/0.14.4/0.16.9, flash-attn==2.8.3, CUDA-pinned torch |
| `decisionintelligence/CATCH` | HEAD | REQUIRES_REPO2ROCM | high | torch.backends.cudnn flags without ROCm guard in ts_benchmark/utils/random_utils.py:26-28 |
| `DeepakTatachar/CSL-Mem` | HEAD | REQUIRES_REPO2ROCM | high | are the hardcoded CUDA assumptions that would cause errors on ROCm without proper guards. |
| `dmis-lab/TemporalHead` | HEAD | REQUIRES_REPO2ROCM | high | hardcoded device='cuda' in source code, unguarded torch.backends.cudnn usage |
| `dxlong2000/NLPromptEval` | HEAD | REQUIRES_REPO2ROCM | high | bitsandbytes (CUDA-only quantization), deepspeed (CUDA-only distributed training), distuti |
| `facebookresearch/luckmatters` | HEAD | REQUIRES_REPO2ROCM | high | CUDA-specific torch version pins, syntax error in Python code, torch-geometric packages wi |
| `FFY0/AdaKV` | 04497aba | REQUIRES_REPO2ROCM | high | flash_attn dependency, custom CUDA extension compilation |
| `flairNLP/VerbalizED` | HEAD | REQUIRES_REPO2ROCM | high | hardcoded cuda:0 device, nvidia-smi calls, undeclared dependencies (bs4, requests, tqdm) |
| `FYYFU/HeadKV` | 0862a095 | REQUIRES_REPO2ROCM | high | flash_attn dependency lacks ROCm wheel, requires AMD Triton backend installation |
| `Graph-COM/LayerDAG` | HEAD | REQUIRES_REPO2ROCM | high | CUDA-pinned torch==1.12.0+cu116 and dgl==1.1.0+cu116 installations, torch.backends.cudnn u |
| `GuanchengWan/EARTH` | HEAD | REQUIRES_REPO2ROCM | high | are: |
| `kaistAI/Knowledge-Entropy` | HEAD | REQUIRES_REPO2ROCM | high | hardcoded nccl backend, flash_attn dependency, undeclared dependencies (olmo_core, botocor |
| `Kambm/convolutional_diffusion` | HEAD | REQUIRES_REPO2ROCM | high | syntax error in scales_calibration.py:95 |
| `LehengTHU/AlphaRec` | HEAD | REQUIRES_REPO2ROCM | high | torch.backends.cudnn usage without HIP guard, Python 3.12 incompatible version pins, distu |
| `mint-vu/MCNC` | HEAD | REQUIRES_REPO2ROCM | medium | nccl-distributed only, no dependency manifest |
| `ModelTC/HarmoniCa` | HEAD | REQUIRES_REPO2ROCM | high | are: |
| `nanomaoli/llm_reproducibility` | HEAD | REQUIRES_REPO2ROCM | high | for AMD/ROCm compatibility: |
| `neuro-galaxy/torch_brain` | HEAD | REQUIRES_REPO2ROCM | medium | xformers dependency, torch.backends.cudnn calls without HIP guard |
| `shengliu66/VTI` | HEAD | REQUIRES_REPO2ROCM | high | bitsandbytes==0.41.0 (CUDA-only), missing imports (requests, BytesIO), missing llava depen |
| `showmeon/TimeEmb` | HEAD | REQUIRES_REPO2ROCM | medium | hardcoded .cuda() calls in AutoCorrelation.py, missing ptflops dependency, deprecated torc |
| `SongW-SW/CEB` | 436943b7 | REQUIRES_REPO2ROCM | high | vllm dependency (CUDA-only PyPI package) |
| `tanghme0w/ACL25-CoPE` | HEAD | REQUIRES_REPO2ROCM | medium | nccl-distributed only, deprecated torch.cuda.amp usage, undeclared orjson |
| `THU-MIG/PrefixKV` | HEAD | REQUIRES_REPO2ROCM | high | bitsandbytes in requirements.txt (CUDA-only package) |
| `thunlp/FR-Spec` | 29d0136b | REQUIRES_REPO2ROCM | high | custom CUDA extensions, flash_attn dependency, extensive CUDA/C++ codebase |
| `togethercomputer/moa` | 1b5cab0f | REQUIRES_REPO2ROCM | high | flash_attn, bitsandbytes, vllm, xformers |
| `UAlberta-NLP/SemEval2025-EAMT` | HEAD | REQUIRES_REPO2ROCM | high | nvidia |
| `XiaoAI1989/ORMind` | HEAD | REQUIRES_REPO2ROCM | high | syntax errors in dataset files, undeclared gurobipy dependency |
| `xihongyang1999/AIRMVC` | HEAD | REQUIRES_REPO2ROCM | high | no actual source code to run |
| `xinyuluo8561/Stacey` | HEAD | REQUIRES_REPO2ROCM | medium | no dependency manifest, undeclared torch / fragile imports |
| `xlang-ai/BRIGHT` | HEAD | REQUIRES_REPO2ROCM | medium | risky old version pins (cohere==4.36, transformers==4.40) prone to API drift |
| `yjucho1/CoMRes` | HEAD | REQUIRES_REPO2ROCM | high | missing __init__.py files in layers/, models/, utils/, exp/ directories causing import fai |
| `YZY-stack/Effort-AIGI-Detection` | HEAD | REQUIRES_REPO2ROCM | high | CUDA-pinned PyTorch in install.sh, torch.cuda.amp deprecated usage, unguarded cudnn calls |
| `ZhiningLiu1998/SelfElicit` | 2fa9d1d3 | REQUIRES_REPO2ROCM | high | hardcoded CUDA assertion in utils.py:28, torch.cuda.empty_cache() in self_elicit.py:373, C |
| `zhoujiahuan1991/NeurIPS2025-KFF` | HEAD | REQUIRES_REPO2ROCM | high | CUDA-pinned PyTorch installation, torch.backends.cudnn flags without HIP guard, missing ro |
