from elasticsearch import Elasticsearch

class ElasticSearchClient:
    def __init__(self, elastic_url, username='HDServer', password='d5Dft73gtI8*'):
        self.client = Elasticsearch(
            [elastic_url],
            basic_auth=(username, password)
        )

    def get_document(self, index, doc_id):
        """
        Retrieves a document by its ID from the specified index.
        """
        try:
            response = self.client.get(index=index, id=doc_id)
            return response['_source']
        except Exception as e:
            print(f"Error retrieving document: {e}")
            return None

    def search_documents(self, index, query, sort=None, size=10):
        """
        Searches for documents in the specified index using the provided query.
        """
        try:
            body = {"query": query}
            if sort:
                body["sort"] = sort
            response = self.client.search(index=index, body=body, size=size)
            return response['hits']['hits']
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def update_document(self, index, doc_id, update_body):
        """
        Updates a document by its ID in the specified index with the provided update body.
        """
        try:
            response = self.client.update(index=index, id=doc_id, body={"doc": update_body})
            return response
        except Exception as e:
            print(f"Error updating document: {e}")
            return None