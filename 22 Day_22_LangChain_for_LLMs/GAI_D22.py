import os
os.environ['OPENAI_API_KEY'] = ""

import streamlit as st
# Set the title using StreamLit
st.title(' My Chatbot ')
input_text = st.text_input('Enter Your Text: ') 


from langchain.prompts import PromptTemplate

title_template = PromptTemplate(
	input_variables = ['concept'], 
	template='Give me a youtube video title about {concept}'
)

script_template = PromptTemplate(
	input_variables = ['title', 'wikipedia_research'], 
	template='''Give me an attractive youtube video script based on the title {title} 
	while making use of the information and knowledge obtained from the Wikipedia research:{wikipedia_research}'''
)


from langchain.memory import ConversationBufferMemory

memoryT = ConversationBufferMemory(input_key='concept', memory_key='chat_history')
memoryS = ConversationBufferMemory(input_key='title', memory_key='chat_history')


from langchain.llms import OpenAI

model = OpenAI(temperature=0.6) 

from langchain.chains import LLMChain
chainT = LLMChain(llm=model, prompt=title_template, verbose=True, output_key='title', memory=memoryT)
chainS = LLMChain(llm=model, prompt=script_template, verbose=True, output_key='script', memory=memoryS)


from langchain.utilities import WikipediaAPIWrapper
wikipedia = WikipediaAPIWrapper()

if input_text: 
	title = chainT.run(input_text)
	wikipedia_research = wikipedia.run(input_text) 
	script = chainS.run(title=title, wikipedia_research=wikipedia_research)

	st.write(title) 
	st.write(script) 

	with st.expander('Wikipedia-based exploration: '): 
		st.info(wikipedia_research)
