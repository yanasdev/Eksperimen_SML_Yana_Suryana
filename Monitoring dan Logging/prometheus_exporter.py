import time
from prometheus_client import Counter, Gauge, start_http_server

REQUEST_COUNT = Counter('model_requests_total', 'Total request count to model serving')
REQUEST_LATENCY = Gauge('model_request_latency_seconds', 'Latest request latency')
MODEL_SCORE = Gauge('model_prediction_score', 'Latest model prediction score')

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        time.sleep(1)
