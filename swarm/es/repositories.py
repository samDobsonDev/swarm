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

    def find_user_by_channel_and_id(self, company, brand, channel, id):
        """
        Finds a user by the specified channel and ID.
        """
        index = f"{company}_{brand}_end_user_data"
        query = {
            "bool": {
                "must": [
                    {"term": {"type.keyword": "chat_user"}},
                    {f"term": {f"channelIds.{channel}.keyword": id}}
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

    def verify_user(self, company, brand, user_id, new_email, channel):
        """
        Verifies a user by updating their email and verifiedChannels.
        """
        index = f"{company}_{brand}_end_user_data"
        user = self.find_user_by_channel_and_id(company, brand, channel, user_id)

        if user:
            # Update the email and verifiedChannels
            verified_channels = user.get("verifiedChannels", {})
            verified_channels[channel] = {"verifiedOn": datetime.now().isoformat()}

            update_body = {
                "email": new_email,
                "verifiedChannels": verified_channels
            }

            self.elastic_search_client.update_document(
                index=index,
                doc_id=user["doc_id"],
                update_body=update_body
            )
            return update_body
        return None

class OrderRepository:
    def __init__(self, elastic_search_client):
        self.elastic_search_client = elastic_search_client

    def get_users_latest_order(self, company, brand, email):
        """
        Returns the latest order for a user by their email, extracting relevant details.
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

        if not documents:
            return None

        # Extract relevant fields from the latest order document
        latest_order_doc = documents[0]
        source = latest_order_doc.get('_source', {})
        order_info = {
            "orderState": source.get("orderState"),
            "orderNumber": source.get("orderNumber"),
            "orderDateTime": source.get("orderDateTime"),
            "totalIncVat": source.get("totalIncVat"),
            "orderCurrency": source.get("orderCurrency"),
            "orderItems": source.get("orderItems"),
            "payments": source.get("payments")
        }

        return order_info

    def get_shipment_details_by_order_number(self, company, brand, email, order_number):
        """
        Retrieves shipment details for a given order number and email.
        """
        index = f"{company}_{brand}_hdorder"
        # Check if the order exists
        order_query = {
            "bool": {
                "must": [
                    {"term": {"customerEmail.keyword": email}},
                    {"term": {"orderNumber.keyword": order_number}}
                ]
            }
        }
        order_documents = self.elastic_search_client.search_documents(index, order_query)

        if not order_documents:
            return None

        # Find matching shipping items
        shipping_query = {
            "bool": {
                "must": [
                    {"term": {"orderNumber.keyword": order_number}},
                    {"term": {"type.keyword": "shipping_item"}}
                ]
            }
        }
        shipping_documents = self.elastic_search_client.search_documents(index, shipping_query)

        if not shipping_documents:
            return None

        # Extract relevant fields from each document
        shipment_details = []
        for doc in shipping_documents:
            source = doc.get('_source', {})
            shipment_info = {
                "trackingUrl": source.get("trackingUrl"),
                "eta": source.get("eta"),
                "orderNumber": source.get("orderNumber"),
                "trackingNumber": source.get("trackingNumber"),
                "address": {
                    "fullName": source.get("shipToFullName"),
                    "addressLine1": source.get("shipToAddressLine1"),
                    "city": source.get("shipToCity"),
                    "postcode": source.get("shipToPostcode"),
                    "country": source.get("shipToCountry")
                },
                "carrier": source.get("carrierShippingMethodCode"),
                "shippingMethod": source.get("shippingMethod", {}).get("type")
            }
            shipment_details.append(shipment_info)

        return shipment_details if shipment_details else None