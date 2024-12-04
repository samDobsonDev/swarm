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

# Global variable to store return eligibility
return_eligibility = None

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
        return f"Stock alert added."
    return "Failed to add stock alert."

def remove_stock_alert_tool(product_id: str):
    """
    Removes a stock alert for the user
    """
    stock_alerts = user_repo.remove_stock_alert(company, brand, email, product_id)
    if stock_alerts is not None:
        return f"Stock alert removed."
    return "Failed to remove stock alert."

def format_order_details(order_info):
    """
    Formats the order details into a natural language description.
    """
    order_state = order_info['orderState']
    order_status = {
        "INCOMPLETE": "The order is being prepared for shipment.",
        "COMPLETE": "The order has been delivered.",
        # Add more states as needed
    }.get(order_state, f"The order is currently in {order_state} state.")

    formatted_order = (
        f"Order number {order_info['orderNumber']} was placed on {order_info['orderDateTime']}. "
        f"The total amount for this order is {order_info['totalIncVat']} {order_info['orderCurrency']}, "
        f"including VAT at a rate of {order_info['vatRate']}%. "
        f"The order is under the name of {order_info['customerFullName']} and will be sent to the email {order_info['customerEmail']}. "
        f"{order_status} "
        f"The order includes the following items: "
    )

    for item in order_info['orderItems']:
        formatted_order += (
            f"{item['itemDescription1']} in size {item['itemSize']} and color {item['itemColour']}, "
            f"which costs {item['unitCostIncVat']} {order_info['orderCurrency']}. "
        )

    formatted_order += "The payment details are as follows: "
    for payment in order_info['payments']:
        formatted_order += (
            f"A payment of {payment['chargeAmount']} was made using {payment['paymentType']}, "
            f"with the reference {payment['paymentReference']}. "
        )

    return formatted_order

def get_latest_order_tool():
    """
    Retrieves information regarding the user's latest order.
    """
    latest_order = order_repo.get_users_latest_order(company, brand, email)
    if latest_order:
        return f"Order details: {latest_order}"
    return "No orders found."

def get_shipment_details_tool(order_number: str):
    """
    Retrieves shipment details for a given order number.
    """
    shipment_details = order_repo.get_shipment_details_by_order_number(company, brand, email, order_number)
    if shipment_details:
        return f"Shipment details: {shipment_details}"
    return "No shipment details found for the given order number."

def generate_verification_pin_tool(provided_email: str):
    """
    Generates a verification PIN for the user and sends it to the email address they wish to verify with.
    """
    # In a real scenario, you would generate a random PIN and send it to the user's email.
    # Here, we are using a fixed PIN for demonstration purposes, neither are we sending an actual email.
    return f"A verification PIN has been sent to {provided_email}. Please use the PIN 1234 to verify."

def verify_user_tool(email: str, provided_pin: str):
    """
    Verifies the user by comparing the provided PIN with the expected PIN.
    """
    if provided_pin == pin:
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
    user = user_repo.find_user_by_channel_and_id(company, brand, channel, user_id)
    if user:
        verified_status = "Verified" if user.get('verifiedChannels') else "Not Verified"
        return f"The user is {verified_status}."
    return "User details not found."

def check_return_eligibility_tool(order_number: str):
    """
    Checks if an order is eligible for return, exchange, or refund.
    """
    global return_eligibility
    return_eligibility = True
    return f"Order {order_number} is eligible for exchange or refund."


def get_return_methods_tool(order_number: str):
    """
    Returns possible return methods for an order.
    """
    if not return_eligibility:
        return f"Order {order_number} is not eligible for return, exchange, or refund."

    return_methods = ["post", "parcel locker", "in-store drop off"]
    return f"Available return methods for order {order_number}: {', '.join(return_methods)}."


def generate_post_label_tool(order_number: str, postage_class: str):
    """
    Generates a code for printing a return label for post.
    """
    if not return_eligibility:
        return f"Order {order_number} is not eligible for return, exchange, or refund."

    return f"Use code POST1234 to print your return label for order {order_number} and return it to us via {postage_class}."


def locate_parcel_lockers_tool():
    """
    Locates nearby parcel lockers.
    """
    if not return_eligibility:
        return "Order is not eligible for return, exchange, or refund."

    lockers = ["Locker A - 123 Main St", "Locker B - 456 Elm St"]
    return f"Nearby parcel lockers: {', '.join(lockers)}."


def generate_parcel_locker_code_tool(locker_choice: str):
    """
    Generates a code for a chosen parcel locker.
    """
    if not return_eligibility:
        return "Order is not eligible for return, exchange, or refund."

    return f"Use code LOCKER5678 at {locker_choice} to open the locker."

def locate_nearest_stores_tool():
    """
    Locates nearest stores and their opening times.
    """
    stores = [
        "Store A - 123 Main St, Open 9am-9pm",
        "Store B - 456 Elm St, Open 10am-8pm"
    ]
    return f"Nearest stores: {', '.join(stores)}."

def process_refund_tool(order_number: str):
    """
    Processes a refund for the given order number.
    """
    if not return_eligibility:
        return f"Order {order_number} is not eligible for a refund."

    # Simulate refund processing
    return f"Refund for order {order_number} will be processed when we receive the returned item/s. Once we have received said item/s, the refund will take 3-5 working days to appear in your account."

def pick_another_item_tool(order_number: str, item_choice: str):
    """
    Allows the customer to pick another item for exchange.
    """
    if not return_eligibility:
        return f"Order {order_number} is not eligible for an exchange."

    # Simulate item exchange process
    return f"Item {item_choice} has been selected for exchange for order {order_number}. Once we have received the returned item/s, we will send your new item/s out to you."