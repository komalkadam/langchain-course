from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from tavily import TavilySearch

tavily_client = TavilyClient())

@tool
def search_tool(query: str) -> str:
    """
    Tool that Searches over internet for the given query and returns the results.
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    print(f"Searching for: {query}")
    # Placeholder for search functionality
    return tavily_client.search(query)

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7)
#tools = [search_tool]
tools =[TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="Search for the latest job openings in Pune for tech lead positions on LinkedIn and provide a summary of the top 5 results.")]})
    print(f"Agent response: {result}")


if __name__ == "__main__":
    main()
