import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

def generate_recipe(ingredients, feedback=None):
    recipe_template = PromptTemplate(
        input_variables=['ingredients'],
        template='Please write a detailed recipe based on the provided list of ingredients: {ingredients}'
    )

    # Memory
    recipe_memory = ConversationBufferMemory(input_key='ingredients', memory_key='chat_history')

    # Language Model
    llm = OpenAI(temperature=0.9)

    # Recipe Generation Chain
    recipe_chain = LLMChain(llm=llm, prompt=recipe_template, output_key='recipe', memory=recipe_memory)

    # Generate the recipe
    recipe = recipe_chain.run(ingredients=ingredients)

    return recipe


def main():
    st.title('AutoGPT Recipe Generator')

    # User input
    ingredients = st.text_input('Enter the ingredients separated by commas')

    # Recipe output
    recipe_output = st.empty()

    if st.button('Generate Recipe') and ingredients:
        with st.spinner('Generating recipe...'):
            try:
                # Generate the recipe
                recipe = generate_recipe(ingredients)

                # Display the generated recipe
                recipe_output.write(recipe)
        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    # Set the OpenAI API key
    os.environ['OPENAI_API_KEY'] = apikey

    # Run the main function
    main()
