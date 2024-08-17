import json
import os
from datetime import datetime
from typing import Any, Dict
from dotenv import load_dotenv
import threading

from langchain.agents import AgentExecutor, Tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_astradb import AstraDBVectorStore
# from langchain.tools import WikipediaQueryRun
# from langchain.utilities import WikipediaAPIWrapper
import uuid  # To generate unique conversation IDs
import sqlite3
from utils.location import MallLocator  # Import the MallLocator class
from utils.parking import ParkingDataSimulation

load_dotenv('/Users/azure/Desktop/ZJ/.env')

# Azure OpenAI configuration
AZURE_OPENAI_CHAT_DEPLOYMENT = "agentgpt4o"
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
AZURE_OPENAI_EMBEDDING_API_VERSION = "2023-05-15"

# Database and file paths
db_file = '../data/IoT/parking_data.db'
conversation_file = '../data/conversation/chatbot.json'

# Initialize the MallLocator
locator = MallLocator()

# Initialize variables to hold conversation context
current_conversation_id = None
current_location = None
current_lat = None
current_lon = None

# 实例化停车数据模拟类
parking_simulation = ParkingDataSimulation(db_file)

# 启动停车数据模拟
parking_thread = threading.Thread(target=parking_simulation.start_simulation, daemon=True)
parking_thread.start()

# 实例化商场位置定位类
mall_locator = MallLocator()

# Define tool for querying parking data
def parking_query_tool(query: str) -> str:
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute('SELECT MAX(timestamp) FROM parking_data')
        latest_timestamp = cursor.fetchone()[0]

        if not latest_timestamp:
            return "目前没有可用的停车数据。"

        cursor.execute('''
            SELECT level, area, available_spots
            FROM parking_data
            WHERE timestamp = ?
            ORDER BY level, area
        ''', (latest_timestamp,))

        results = cursor.fetchall()

        if results:
            return "\n".join(f"在 {level} 层的 {area} 有 {available_spots} 个空闲车位。" for level, area, available_spots in results)
        else:
            return "目前没有空闲车位。"
    except Exception as e:
        return f"查询停车数据时发生错误: {str(e)}"
    finally:
        if conn:
            conn.close()

# Define tool for querying vector database
def vector_db_query(query: str) -> str:
    try:
        results = vstore.similarity_search(query, k=1)
        if results:
            return results[0].page_content
        return "没有找到相关信息。"
    except Exception as e:
        return f"查询向量数据库时发生错误: {str(e)}"

# Define tool to get a random location
def location_tool(query: str = "") -> str:
    global current_location, current_lat, current_lon
    if not current_location:
        current_location, current_lat, current_lon = locator.get_random_location()
    return f"Location: {current_location}, Coordinates: ({current_lat}, {current_lon})"

# Initialize tools
tools = [
    Tool(name="ParkingQueryTool", func=parking_query_tool, description="用于查询停车场各个区域的空闲车位的工具。"),
    # Tool(name="WikipediaQueryTool", func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run, description="用于查询维基百科上的通用知识信息。"),
    Tool(name="VectorDBQueryTool", func=vector_db_query, description="用于查询向量数据库中的信息。"),
]

# Initialize OpenAI model and vector store
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
)

llm = AzureChatOpenAI(
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    temperature=0,
)

vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name='k110813',
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)

# Define Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是智能客服机器人Luna,代表K11购物艺术中心。你的回答必须解决顾客的问题, 並記錄反饋給商場管理者。请遵循以下指南:

1. 问题解决优先:
   - 快速理解顾客的核心问题
   - 优先使用最相关的方法来解决问题，盡量使用最多的工具
   - 如果一个方法无法完全解决问题,考虑结合多个方法

2. 回答策略:
   - 简洁明了, 必要时提供详细解释
   - 用优雅知性的语气回答
   - 給出鏈結或指引以便顧客進一步了解

- 根据时间地点适当调整回答, 如考虑商场营业时间、活动地点等 当前时间: {current_time}; 当前位置: {location}

请灵活运用这些指南,为顾客提供最佳的问题解决体验。"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Custom conversation memory
class CustomConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str = inputs["input"]
        output_str = outputs["output"]
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chat_history": self.chat_memory.messages
        }

# Initialize memory component
memory = CustomConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

# Initialize agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Function to start a new conversation
def start_new_conversation():
    global current_conversation_id, current_location, current_lat, current_lon
    current_conversation_id = str(uuid.uuid4())  # Generate a unique conversation ID
    current_location, current_lat, current_lon = locator.get_random_location()

# Function to save conversation
def save_conversation(question: str, answer: str):
    timestamp = datetime.now().isoformat()

    conversation = {
        "conversation_id": current_conversation_id,
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "location": {
            "name": current_location,
            "latitude": current_lat,
            "longitude": current_lon
        }
    }
    
    try:
        os.makedirs(os.path.dirname(conversation_file), exist_ok=True)
        file_path = conversation_file
    except Exception:
        file_path = 'chatbot.json'
    
    try:
        conversations = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    conversations = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON data: {e}")
                    conversations = []  # Start with an empty list if the file is corrupted

        conversations.append(conversation)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    except Exception as e:
        print(f"保存对话时发生错误: {str(e)}")

# Function to get a response
def get_response(question: str) -> str:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = location_tool(question)
    
    try:
        response = agent_executor({'input': question, 'current_time': current_time, 'location': location})
        answer = response['output']
        
        save_conversation(question, answer)
        
        return answer
    except Exception as e:
        error_message = f"生成回答时发生错误: {str(e)}"
        save_conversation(question, error_message)
        return error_message

# Main conversation loop
if __name__ == "__main__":
    print("Chat with Luna (type 'bye' to exit):")
    start_new_conversation()  # Start a new conversation

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Luna: 再见, 祝您购物愉快！")
            break
        response = get_response(user_input)
        print(f"Luna: {response}")
