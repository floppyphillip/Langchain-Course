import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

print("Initializing components...")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5")

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}

    Question: {question}

    Provide a detailed answer:
    
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (Langchain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}
     
    """

    retrieval_chain = (
        RunnablePassthrough.assign(
            # context=itemgetter[str] ("question") | retriever | format_docs
            context=itemgetter("question") | retriever | format_docs
        )
         
        
        | prompt_template | llm | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print ("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print('\nAnswer:')
    print(result_with_lcel)
