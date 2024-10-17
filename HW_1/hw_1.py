import os

# import getpass

from langchain.chat_models import GigaChat

from langchain.schema import HumanMessage, SystemMessage

from typing import List, Union


from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.schema import Document


# # 1. GigaChat
# Define GigaChat throw langchain.chat_models


def get_giga(giga_key: str) -> GigaChat:
    return GigaChat(credentials=giga_key, model="GigaChat", verbose=False, timeout=30, verify_ssl_certs=False)


def test_giga():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)
    print(isinstance(giga, GigaChat))


# # 2. Prompting
# ### 2.1 Define classic prompt


# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> List[Union[SystemMessage, HumanMessage]]:
    return [
        SystemMessage(content="You are a helpful chat-bot who answers people's questions."),
        HumanMessage(content=user_content),
    ]


# Let's check how it works
def test_prompt():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)
    user_content = "Hello!"
    prompt = get_prompt(user_content)
    res = giga.invoke(prompt)
    print(res.content)


# ### 3. Define few-shot prompting


# Implement a function to build a few-shot prompt to count even digits in the given number.
# The answer should be in the format 'Answer: The number {number} consist of {text} even digits.',
# for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:
    few_shot_prompt = f"""
    How many even digits are in the number 11223344?
    Answer should be in the following  format: Answer: The number 11223344 consist of four even digits.

    How many even digits are in the number 8664024?
    Answer should be in the following  format: Answer: The number 8664024 consist of seven even digits.

    How many even digits are in the number 13351897?
    Answer should be in the following  format: Answer: The number 13351897 consist of zero even digits.

    How many even digits are in the number {number}?
    """
    return [HumanMessage(content=few_shot_prompt)]


# Let's check how it works
def test_few_shot():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga = get_giga(giga_key)
    number = "62388712774"
    number = "11223344"
    prompt = get_prompt_few_shot(number)
    res = giga.invoke(prompt)
    print(f"Prompt:\n{prompt}\n\nModel answer:\n{res.content}")


# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code to build llama_index
# across your own documents. For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt: str = """
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        self.path_to_data: str = path_to_data
        self.llm = llm

    def query(self, user_prompt: str) -> str:
        embed_model = self.get_embed_model("sentence-transformers/all-mpnet-base-v2")

        service_context = self.get_service_context(llm=self.llm, embed_model=embed_model, chunk_size=1024)

        documents = self.get_documents(self.path_to_data)
        index = self.get_vector_store_index(documents, service_context=service_context)

        query_engine = index.as_query_engine()

        user_input = self.system_prompt + user_prompt
        response = query_engine.query(user_input)
        return response.response

    def get_embed_model(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> LangchainEmbedding:
        return LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))

    def get_service_context(
        self, llm: GigaChat, embed_model: LangchainEmbedding, chunk_size: int = 1024
    ) -> ServiceContext:
        return ServiceContext.from_defaults(chunk_size=chunk_size, llm=llm, embed_model=embed_model)

    def get_documents(self, path_to_data: str = r"\data") -> List[Document]:
        return SimpleDirectoryReader(path_to_data).load_data()

    def get_vector_store_index(self, documents: List[Document], service_context: ServiceContext) -> VectorStoreIndex:
        return VectorStoreIndex.from_documents(documents, service_context=service_context)


# Let's check
def test_llama_index():
    giga_key = os.environ.get("GIGACHAT_CREDENTIALS")
    giga_pro = GigaChat(credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)

    llama_index = LlamaIndex("data/", giga_pro)
    res = llama_index.query("what is attention is all you need?")
    print(res)


if __name__ == "__main__":
    test_giga()
    test_prompt()
    test_few_shot()
    test_llama_index()
