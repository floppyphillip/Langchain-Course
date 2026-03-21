from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

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

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
# tools = [search]
agent = create_agent(model=llm,tools=tools)


def main():
    print("Hello from langchain-search-agent!")
    result = agent.invoke({"messages":HumanMessage(content="What is the wheather in Tokyo")})
    print(result)

if __name__ == "__main__":
    main()
