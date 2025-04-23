from openai import OpenAI
import os
from typing import List, Dict, Any, Optional

class BaseAgent:
    """
    所有Agent的基类，提供基本的对话和记忆功能
    """
    def __init__(
        self,
        role: str,
        system_prompt: str,
        model: str = "deepseek-chat"
    ):
        """
        初始化Agent
        
        Args:
            role: Agent的角色名称
            system_prompt: 系统提示词，定义Agent的行为和知识
            model: 使用的模型名称
        """
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.conversation_history = []
        
        # 添加系统提示到对话历史
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})
    
    def get_response(self, message: str, temperature: float = 0.7) -> str:
        """
        获取模型回复
        
        Args:
            message: 输入的消息
            temperature: 温度参数，控制响应的随机性
            
        Returns:
            模型的回复
        """
        # 添加用户消息到历史
        self.conversation_history.append({"role": "user", "content": message})
        
        # 调用API获取回复
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=temperature
            )
            reply = response.choices[0].message.content
            
            # 添加模型回复到历史
            self.conversation_history.append({"role": "assistant", "content": reply})
            
            return reply
            
        except Exception as e:
            error_message = f"API调用出错: {str(e)}"
            print(error_message)
            return error_message
    
    def add_message_to_history(self, role: str, content: str) -> None:
        """
        手动添加消息到对话历史
        
        Args:
            role: 消息的角色 (system, user, assistant)
            content: 消息内容
        """
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        清除对话历史
        
        Args:
            keep_system_prompt: 是否保留系统提示词
        """
        if keep_system_prompt and self.system_prompt:
            self.conversation_history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.conversation_history = []
            
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        return self.conversation_history 