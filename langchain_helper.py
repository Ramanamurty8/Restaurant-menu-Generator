from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain


llm =ChatGroq(temperature=0,groq_api_key='Your key',model_name='llama-3.1-70b-versatile')

def generate_restaurant_name_and_items(cuisine):
    prompt_template_name = PromptTemplate(input_variables=['cuisine'],
                                          template="I want to open a restaurant for {cuisine} food.Suggest only one fancy name for this.Give only name and dont write any text")
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='restaurant_name')

    prompt_template_items = PromptTemplate(input_variables=['restaurant_name'],
                                           template=""" Suggest some menu items for {restaurant_name}.Return it as comma separated list,dont include any other text apart from the items .No preamble""")

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='menu_items')
    chain = SequentialChain(chains=[name_chain, food_items_chain],
                            input_variables=['cuisine'],
                            output_variables=['restaurant_name', 'menu_items'])

    response=chain.invoke({'cuisine': cuisine})
    return response

if __name__ =="__main__":
    print(generate_restaurant_name_and_items("Italian"))