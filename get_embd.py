import numpy as np
import openai
import pandas as pd
import tiktoken
import time
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
openai.api_key = ""
COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    total={}
    for idx,r in df.iterrows():
        value=get_embedding(r.content)
        time.sleep(1)
        total[idx]=value
    return total

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    max_dim = max([int(float(c)) for c in df.columns if c.strip() not in ["title", "heading"]])
    print(max_dim)
    return {
           (r.content): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame , cur:str) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    count=0
    num=0
    for _, section_index in most_relevant_document_sections:        
        document_section = df.loc[section_index]
        chosen_sections_len += len(document_section.content)
        num_tokens = num_tokens_from_string(document_section.content.replace("\n", " ")+"".join(cur), "gpt2")
        if count>2 or num+num_tokens>1500:
            if len(cur)>3 and num+num_tokens>1500 :
                del cur[0]
            break
            
        chosen_sections.append(": " + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        count+=1
        num+=num_tokens
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    header = """You are a helpful assistant who answer questions. Answer the question as truthfully as possible the provided context. And if you're unsure of the answer, say "Sorry, I don't know". 答案用中文回答。"\n\nContext:\n"""
    print(header + "".join(chosen_sections) + '\nThis is  the conversation so far.\n'+"".join(cur))
    num_tokens = num_tokens_from_string(header +"".join(chosen_sections) + "".join(cur), "gpt2")
    print(num_tokens)
    return header + "".join(chosen_sections) + "".join(cur)

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    messages,
    currentme,
    show_prompt: bool = False,
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df,
        currentme
    )
    
    if show_prompt:
        print(prompt)
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = messages
            )

    return response.choices[0].message.content

def get_embd():
    with open('test-婚俗新风.pdf', 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        num_pages = len(pdf_reader.pages)
        paragraphs = []
        total_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text:
                total_text += text
    text_splitter=CharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=20,
                separator="\n"
            )
    pages = total_text.split('\f')
    all_docs=[]
    for page_content in pages:
        docs = [Document(page_content=page_content)]
        all_docs.extend(docs)
    all_docs = text_splitter.split_documents(all_docs)
    paragraphs=[]
    for document in all_docs:
        page_content = document.page_content
        paragraphs.append(page_content)
    df = pd.DataFrame({'content': paragraphs})
    document_embeddings = compute_doc_embeddings(df)
    return df, document_embeddings

#document_embeddings=load_embeddings('embedding-婚俗新风.csv')

#print(document_embeddings)