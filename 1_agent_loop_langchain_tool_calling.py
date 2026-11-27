from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product_name: str) -> float:
    """
    Look up the price of a prodcut in the catalog.
    """
    # Simulate a product price lookup
    product_prices = {
        "laptop": "$999",
        "smartphone": "$699",
        "headphones": "$199",
    }
    print(f"Looking up price for product: {product_name}")
    return product_prices.get(product_name.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price based on the discount tier and return the discounted price.
    Available discount tiers: silver (5%), gold (10%), platinum (15%).
    """
    discount_rates = {
        "silver": 0.05,
        "gold": 0.10,
        "platinum": 0.15,
    }
    discount_rate = discount_rates.get(discount_tier.lower(), 0.0)
    discounted_price = price * (1 - discount_rate)
    print(f"Applying {discount_tier} discount: {discount_rate*100}% off")
    return discounted_price

def run_agent(question: str):
    agent_tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in agent_tools}

    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(agent_tools)

    print(f"Running agent for question: {question}")
    messages = [
        SystemMessage(content=
                      "You are a helpful assistant that can look up product prices and apply discounts."
                      "You have access to the following tools: get_product_price and apply_discount."
                      "You should only use the tools provided."
                      "STRICT RULES: You must follow these exactly. Do not make up any prices or discounts. If you don't know the answer, say 'I don't know'."
                      "You must call get_product_price first to get the price of the product, and then call apply_discount to apply the discount."
                      "You must not make up any prices or discounts. If you don't know the answer, say 'I don't know'."
                      "Never calculate discounts manually. Always use the apply_discount tool to calculate the discounted price."
                      "If user does not specify discount tier ask them to specify one of the following: silver, gold, platinum. do not assume any discount tier. If user does not specify a product name, ask them to specify one of the following: laptop, smartphone, headphones. do not assume any product name."
                      ),
                      
                      HumanMessage(content=question)] 

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}:")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("No tool calls made. Final response:")
            print(ai_message.content)
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool call: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")

        obeservation = tool_to_use.invoke(**tool_args)
        print(f"Tool '{tool_name}' returned: {obeservation}")
        messages.append(ai_message)  # Append the AI message to the conversation
        messages.append(ToolMessage(content=str(obeservation), tool_name=tool_name, tool_call_id=tool_call_id))

    print("ERROR - Max iterations reached. Final response:")
    print(messages[-1].content)
    return None
        

if __name__ == "__main__":
    print("Hello from langchain-course!")
    question = "What is the price of a laptop with a gold discount?"
    run_agent(question)