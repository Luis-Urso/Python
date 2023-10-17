## Import Pinecone and Initialize it

print('1 - Initializing Pinecone Connection...')

import pinecone

# connect to pinecone environment
pinecone.init(
    api_key="676be22d-86c1-48cd-af60-688993fe5fc3",
    environment="asia-southeast1-gcp-free"
)

print('Done.')

## Initialize the Pinecone Index 

print('2 - Initializing the Pinecone Index...')

index_name = "abstractive-question-answering"

# check if the abstractive-question-answering index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension=768,
        metric="cosine"
    )

# connect to abstractive-question-answering index we created
index = pinecone.Index(index_name)

print('Done.')

## Define the Retriever

print('3 - Initializing the Retriever...')

import torch
from sentence_transformers import SentenceTransformer

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on: ",device)
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)
print(retriever)

print('Done.')

## Initialize the Generator

print('4 - Initialize the Generator...')

from transformers import BartTokenizer, BartForConditionalGeneration

# load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)

print('Done.')

## Defining Query Helper Functions 

print('5 - Defining Query Helper Functions...')

def query_pinecone(query, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
    # extract passage_text from Pinecone search result and add the <P> tag
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query

print('Done.')


## Generating Answers

print('6 - Ready for Questions...')

from pprint import pprint

def generate_answer(query):
    # tokenize the query to get input_ids
    inputs = tokenizer([query], max_length=1024, return_tensors="pt")
    # use generator to predict output ids
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=40)
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return pprint(answer)

## Query loop

print('Make your questions below. To leave type exit or crtl-c.')
query=''
while query!='exit':
    query = input("Ask me something ?")
    if query!='exit':
        context = query_pinecone(query, top_k=5)
        query = format_query(query, context["matches"])
        generate_answer(query)