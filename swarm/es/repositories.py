from datetime import datetime
from swarm.es.elasticsearch_client import ElasticSearchClient

class UserRepository:
    def __init__(self, elastic_search_client: ElasticSearchClient):
        self.elastic_search_client = elastic_search_client

    def find_user_by_email(self, company, brand, email):
        """
        Finds all users with the specified email.
        """
        index = f"{company}_{brand}_end_user_data"
        query = {
            "bool": {
                "must": [
                    {"term": {"type.keyword": "chat_user"}},
                    {"term": {"email.keyword": email}}
                ]
            }
        }
        documents = self.elastic_search_client.search_documents(index, query)
        if documents:
            doc = documents[0]
            user = doc.get('_source', {})
            return {
                "doc_id": doc.get('_id'),
                "firstName": user.get("firstName"),
                "lastName": user.get("lastName"),
                "email": user.get("email"),
                "stockAlerts": user.get("stockAlerts"),
                "verifiedChannels": user.get("verifiedChannels")
            }
        return None

    def get_stock_alerts(self, company, brand, email):
        """
        Retrieves stock alerts for a user by their email.
        """
        user = self.find_user_by_email(company, brand, email)
        if user:
            return user.get("stockAlerts", [])
        return None

    def add_stock_alert(self, company, brand, email, product_id):
        """
        Adds a new stock alert for the user by their email and product ID.
        """
        user = self.find_user_by_email(company, brand, email)
        if user:
            stock_alerts = user.get("stockAlerts", [])
            new_alert = {"productId": product_id, "timestamp": datetime.now().isoformat()}
            stock_alerts.append(new_alert)
            self.elastic_search_client.update_document(
                index=f"{company}_{brand}_end_user_data",
                doc_id=user["doc_id"],
                update_body={"stockAlerts": stock_alerts}
            )
            return stock_alerts
        return None

    def remove_stock_alert(self, company, brand, email, product_id):
        """
        Removes the specified stock alert for the user by their email and product ID.
        """
        user = self.find_user_by_email(company, brand, email)
        if user:
            stock_alerts = user.get("stockAlerts", [])
            stock_alerts = [alert for alert in stock_alerts if alert["productId"] != product_id]
            self.elastic_search_client.update_document(
                index=f"{company}_{brand}_end_user_data",
                doc_id=user["doc_id"],
                update_body={"stockAlerts": stock_alerts}
            )
            return stock_alerts
        return None

class OrderRepository:
    def __init__(self, elastic_search_client):
        self.elastic_search_client = elastic_search_client

    def find_completed_orders_by_email(self, company, brand, email):
        """
        Finds completed orders for a user by their email.
        """
        index = f"{company}_{brand}_hdorder"
        query = {
            "bool": {
                "must": [
                    {"term": {"customerEmail.keyword": email}},
                    {"term": {"orderState.keyword": "COMPLETE"}}
                ]
            }
        }
        return self.elastic_search_client.search_documents(index, query)

    def find_incomplete_orders_by_email(self, company, brand, email):
        """
        Finds incomplete orders for a user by their email.
        """
        index = f"{company}_{brand}_hdorder"
        query = {
            "bool": {
                "must": [
                    {"term": {"customerEmail.keyword": email}},
                    {"term": {"orderState.keyword": "INCOMPLETE"}}
                ]
            }
        }
        return self.elastic_search_client.search_documents(index, query)

    def get_users_latest_order(self, company, brand, email):
        """
        Returns the latest order for a user by their email, regardless of order state.
        """
        index = f"{company}_{brand}_hdorder"
        query = {
            "bool": {
                "must": [
                    {"term": {"customerEmail.keyword": email}}
                ]
            }
        }
        # Sort by orderDateTime in descending order to get the latest order
        sort = [{"orderDateTime": {"order": "desc"}}]
        documents = self.elastic_search_client.search_documents(index, query, sort=sort, size=1)
        return documents[0] if documents else None