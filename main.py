from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

# from tavily import TavilyClient

load_dotenv()


# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result

#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
# tools = [search]
agent = create_agent(model=llm,tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-search-agent!")
    result = agent.invoke(
                {
                "messages":HumanMessage(
                    content="Search for 3 job openings fo an ai engineer using langchain in the bay area on linkedin"
                    )
                }
            )
    print(result)

if __name__ == "__main__":
    main()
