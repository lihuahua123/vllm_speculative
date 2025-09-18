# Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving

Extend base on vLLM (https://github.com/vllm-project).
## Environment
```
VLLM_USE_PRECOMPILED=1 pip install -v --editable .
pip install pandas  joblib  pulp  scikit-learn datasets seaborn python-multipart torch-scatter
```
## dynamic trace
```
bash test_dynamic.sh
```
## static request rate
```
bash test_static.sh
```
