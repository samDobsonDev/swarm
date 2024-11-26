# Initialize a shared ElasticSearchClient instance
from swarm.es.elasticsearch_client import ElasticSearchClient
from swarm.es.repositories import UserRepository, OrderRepository

es_client = ElasticSearchClient(elastic_url = "http://54.154.188.102:9200")

# Initialize repositories
user_repo = UserRepository(es_client)
order_repo = OrderRepository(es_client)

email = "sam.dobsonn@gmail.com"
company = "footasylum"
brand = "footasylum"
pin = "1234"
user_id = "hbgkuHM7TaciF74+gkTwVSa5hr0="
channel = "MESSENGER"

def get_stock_alerts_tool():
    """
    Retrieves the user's active stock alerts
    """
    stock_alerts = user_repo.get_stock_alerts(company, brand, email)
    if stock_alerts:
        return f"Stock alerts: {stock_alerts}"
    return "The user doesn't have any stock alerts set up"

def add_stock_alert_tool(product_id: str):
    """
    Adds a stock alert for the user
    """
    stock_alerts = user_repo.add_stock_alert(company, brand, email, product_id)
    if stock_alerts is not None:
        return f"Stock alert added. Current alerts: {stock_alerts}"
    return "Failed to add stock alert."

def remove_stock_alert_tool(product_id: str):
    """
    Removes a stock alert for the user
    """
    stock_alerts = user_repo.remove_stock_alert(company, brand, email, product_id)
    if stock_alerts is not None:
        return f"Stock alert removed. Current alerts: {stock_alerts}"
    return "Failed to remove stock alert."

def get_completed_orders_tool():
    """
    Retrieves the user's completed orders
    """
    completed_orders = order_repo.find_completed_orders_by_email(company, brand, email)
    if completed_orders:
        return f"Completed orders: {completed_orders}"
    return "No completed orders found."

def get_incomplete_orders_tool():
    """
    Retrieves the user's incomplete orders
    """
    incomplete_orders = order_repo.find_incomplete_orders_by_email(company, brand, email)
    if incomplete_orders:
        return f"Incomplete orders: {incomplete_orders}"
    return "No incomplete orders found."


def get_latest_order_tool():
    """
    Retrieves the user's latest order
    """
    latest_order = order_repo.get_users_latest_order(company, brand, email)
    if latest_order:
        order_info = latest_order["_source"]
        formatted_order = (
            f"Order Number: {order_info['orderNumber']}\n"
            f"Order Date: {order_info['orderDateTime']}\n"
            f"Order State: {order_info['orderState']}\n"
            f"Currency: {order_info['orderCurrency']}\n"
            f"Total (Excluding VAT): {order_info['totalExVAT']}\n"
            f"Total (Including VAT): {order_info['totalIncVat']}\n"
            f"VAT Rate: {order_info['vatRate']}%\n"
            f"Customer: {order_info['customerFullName']} ({order_info['customerEmail']})\n"
            f"Items:\n"
        )

        for item in order_info['orderItems']:
            formatted_order += (
                f"  - {item['itemDescription1']} (Size: {item['itemSize']}, "
                f"Colour: {item['itemColour']}, State: {item['itemState']}, "
                f"Cost: {item['unitCostIncVat']}{order_info['orderCurrency']})\n"
            )

        formatted_order += "Payments:\n"
        for payment in order_info['payments']:
            formatted_order += (
                f"  - Type: {payment['paymentType']}, Amount: {payment['chargeAmount']}, "
                f"Reference: {payment['paymentReference']}\n"
            )

        return formatted_order

    return "No orders found."

def generate_verification_pin_tool(provided_email: str):
    """
    Generates a verification PIN for the user.
    """
    # In a real scenario, you would generate a random PIN and send it to the user's email.
    # Here, we are using a fixed PIN for demonstration purposes, neither are we sending an actual email.
    return f"A verification PIN has been sent to {provided_email}. Please use the PIN 1234 to verify."

def verify_user_tool(email: str, provided_pin: str):
    """
    Verifies the user by comparing the provided PIN with the expected PIN.
    """
    if provided_pin == pin:
        # Call the verify_user method from UserRepository
        update_result = user_repo.verify_user(company, brand, user_id, email, channel)
        if update_result:
            return f"User with email {email} has been successfully verified and updated."
        else:
            return f"User with email {email} could not be updated in the system."
    return f"Verification failed for user with email {email}. Incorrect PIN."

def check_verification_status_tool():
    """
    Checks if the user is verified.
    """
    # Assuming the user context is available and contains the necessary information
    user = user_repo.find_user_by_channel_and_id(company, brand, channel, user_id)
    if user:
        verified_status = "Verified" if user.get('verifiedChannels') else "Not Verified"
        return f"The user is {verified_status}."
    return "User details not found."