import os
from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
load_dotenv()


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    template = """ Given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                    Your answer shoudl contain only a URL"""
    propmt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"])
    tool_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get Linkedin Page URL"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm, tools=tool_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tool_for_agent, verbose=True)

    # Invocing the agent
    result = agent_executor(
        input={"input": propmt_template.format_prompt(name_of_person=name)})
    linkedin_profile_url = result["output"]
    return linkedin_profile_url


if __name__ == "__main__":
    linkedin_url = lookup(name="Eden Marco")
    print(linkedin_url)