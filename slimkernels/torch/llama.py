from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLaMA3Inference:
    def __init__(self, model_name="meta-llama/Llama-2-8b-hf", device="cuda"):
        """
        初始化 LLaMA3 推理类
        
        Args:
            model_name (str): Hugging Face 模型名称
            device (str): 运行设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate(self, prompt, max_length=512, temperature=0.7, top_p=0.95):
        """
        生成文本响应
        
        Args:
            prompt (str): 输入提示
            max_length (int): 最大生成长度
            temperature (float): 采样温度
            top_p (float): 核采样参数
            
        Returns:
            str: 生成的文本响应
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    # 使用示例
    llama = LLaMA3Inference()
    prompt = "请解释量子计算的基本原理。"
    response = llama.generate(prompt)
    print(f"输入: {prompt}")
    print(f"输出: {response}")

if __name__ == "__main__":
    main()
