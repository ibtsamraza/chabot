from concurrent.futures import ThreadPoolExecutor
import threading
import atexit
import time
from threading import Thread

# Initialize the ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers based on your system's capabilities

def create_embedding(files):
    # Dummy function for creating embeddings
    # Implement your file processing logic here
    print("embedding procees is started")
    time.sleep(10)
    return(f"Processing files {files}")
    # Simulate some work
    
def on_completion(future):
    try:
        result = future.result()
        print("Task completed successfully with result:", result)
    except Exception as exc:
        print(f"Task generated an exception: {exc}")

def start_embedding_process():
    # Example file processing and folder path

    files = "hello_worls"
    # Submit the embedding task to the thread pool
    future = executor.submit(create_embedding, files)
    future.add_done_callback(on_completion)
    print("hello world")
    return
    # Optionally handle the result or exceptions from the task
    try:
        result = future.result()  # Waits for the thread to complete and returns the result
        print("Task completed successfully:", result)
    except Exception as exc:
        print(f"Task generated an exception: {exc}")
    # Optionally handle the result or exceptions from the task


# This ensures the executor shuts down properly when the program exits
atexit.register(executor.shutdown)

# Example usage
if __name__ == '__main__':
    start_embedding_process()
