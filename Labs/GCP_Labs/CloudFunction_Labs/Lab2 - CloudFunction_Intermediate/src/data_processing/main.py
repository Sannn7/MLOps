import json
import pandas as pd
from google.cloud import pubsub_v1
from google.cloud import storage

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('mlops-vertexai-project', 'model-training-trigger')
storage_client = storage.Client()

def process_data(request):
    """HTTP triggered function for data processing."""
    request_json = request.get_json()
    bucket_name = request_json.get('bucket', 'mlops-vertexai-project-bucket')
    file_name = request_json.get('file', 'data.csv')
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    file_contents = blob.download_as_text()
    
    from io import StringIO
    data = pd.read_csv(StringIO(file_contents))
    stats = data.describe().to_dict()
    
    processed_data = {
        "status": "data_processed",
        "file": file_name,
        "stats": str(stats)
    }
    
    publisher.publish(topic_path, json.dumps(processed_data).encode('utf-8'))
    print(f"Data processed and published to {topic_path}")
    return json.dumps({"status": "success", "rows": len(data)})
