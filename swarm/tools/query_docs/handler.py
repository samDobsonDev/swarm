from openai import OpenAI
import qdrant_client
import re

client = OpenAI()
qdrant = qdrant_client.QdrantClient(host = 'localhost')
EMBEDDING_MODEL = 'text-embedding-3-large'
collection_name = 'help_center'

def query_qdrant(query, collection, vector_name = 'article', top_k = 5):
    embedded_query = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    ).data[0].embedding
    query_results = qdrant.search(
        collection_name=collection,
        query_vector=(
            vector_name, embedded_query
        ),
        limit=top_k,
    )
    return query_results

def query_docs(query):
    print(f'Searching knowledge base with query: {query}')
    query_results = query_qdrant(query, collection = collection_name)
    output = []
    for i, article in enumerate(query_results):
        title = article.payload["title"]
        text = article.payload["text"]
        url = article.payload["url"]
        output.append((title,text,url))
    if output:
        title, content, _ = output[0]
        response = f"Title: {title}\nContent: {content}"
        truncated_content = re.sub(r'\s+', ' ', content[:50] + '...' if len(content) > 50 else content)
        print('Most relevant article title:', truncated_content)
        return {'response': response}
    else:
        print('no results')
        return {'response': 'No results found.'}
