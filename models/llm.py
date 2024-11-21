from typing import Dict, List, Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

class LLMHandler:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        调用OpenAI API获取回复
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            return content.encode('utf-8').decode('utf-8')
        except Exception as e:
            print(f"调用API时发生错误: {str(e)}")
            raise e 