import os
import warnings
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
import warnings
from few_shots import few_shots
from dotenv import load_dotenv


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")

load_dotenv()


def get_few_shot_db_chain():
    # llm = GooglePalm(google_api_key=api_key, temperature=0.1)
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["api_key"], temperature=0.1)

    db_user = "root"
    db_password = "123456789"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join([example['Question'], example['SQLQuery'], example['Answer']]) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=few_shots)

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=3,
    )

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
    )

    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain


if __name__ == "__main__":
    new_chain = get_few_shot_db_chain()
    print(new_chain.run("How many total t shirts are left in total in stock?"))
