# import os
# import hashlib
# import hmac
# from requests import Session
# from urllib.parse import urljoin

# class BinanceSession(Session):
#     def __init__(self, prefix_url=None, *args, **kwargs):
#         super(BinanceSession, self).__init__(*args, **kwargs)
#         self.prefix_url = prefix_url

#     def request(self, method, url, *args, **kwargs):
#         url = urljoin(self.prefix_url, url)
#         return super(BinanceSession, self).request(method, url, *args, **kwargs)

# secret = os.getenv('BINANCE_SECRET_KEY')
# hashedsig = hashlib.sha256(secret).hexdigest()

# params = urlencode({
#     "signature" : hashedsig,
#     "timestamp" : servertimeint,
# })
# hashedsig = hmac.new(secret.encode('utf-8'), params.encode('utf-8'), hashlib.sha256).hexdigest()

# r = BinanceSession('https://api.binance.com')
# r.headers.update({'': os.getenv('BINANCE_API_KEY')})
# r.params.update({'signature': })
import os
from binance.client import Client
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))