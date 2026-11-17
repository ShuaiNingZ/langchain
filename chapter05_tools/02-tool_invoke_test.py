import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from rich import print as rprint

load_dotenv(override=True)
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
ZHIPUAI_API_BASE_URL = os.getenv("ZHIPUAI_API_BASE_URL")

# 获取大模型
model = init_chat_model(
    model='glm-5.3',
    model_provider="openai",
    api_key=ZHIPUAI_API_KEY,
    base_url=ZHIPUAI_API_BASE_URL
)

# 2. 声明一个函数(工具)
def get_weather(city: str):
    '''获取天气的工具'''
    return f"{city}天气晴朗"

# 3. 将函数绑定在模型上
model_with_tools = model.bind_tools([get_weather])

# 4. 调用模型
response = model_with_tools.invoke("北京的天气怎么样")
rprint(response)