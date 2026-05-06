import pandas as pd

customers = {
    'CustomerID': range(1, 4),
    'Name': ['Bob', 'Rob', 'Kob'],
    'Address': ['Address1', 'Address2', 'Address3']
}
customers = pd.DataFrame(customers)

orders = {
    'OrderID': range(1000, 1004),
    'CustomerID': [1, 2, 1, 5]
}
orders = pd.DataFrame(orders)

result = customers.merge(orders, on='CustomerID', how='left')

print(result)