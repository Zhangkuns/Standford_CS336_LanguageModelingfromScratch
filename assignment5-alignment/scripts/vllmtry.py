from vllm import LLM, SamplingParams
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("miromind-ai/MiroMind-M1-SFT-719K")

# 示例 prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 创建采样参数对象：遇到换行符就停止生成
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

# 创建一个 LLM
llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

# 对 prompts 进行生成。输出是一个 RequestOutput 对象列表，
# 每个对象包含 prompt、生成文本以及其他信息
outputs = llm.generate(prompts, sampling_params)

# 打印输出
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
