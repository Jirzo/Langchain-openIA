import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from third_parties.linkedin import scrape_linkedin_profile
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    print("Hello LangChain")

    summary_template = """
        Given the linkedin information {information} about a person from I want you to create:
        1. A short summary
        2. Two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(input_variables=['information'], template=summary_template)

    # Inicializar una instancia de chatgpr
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # llm = ChatOllama(model="llama3.2")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/", mock=True)

    # Creacion de un chain donde juntaremos el summary template prompt con la instancia de chatgpt
    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": linkedin_data})
    print(res)