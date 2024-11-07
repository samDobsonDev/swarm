def send_email(email_address, message):
  response = f'email sent to: {email_address} with message: {message}'
  return {'response':response}
