import threading
import torch
from collections import deque


class Cluster:
    def __init__(self, model_path):
        self.pool = deque()
        self.model = torch.load(model_path)
        self.lock = threading.Lock()

    def put_data(self):
        csi = input()
        with self.lock:
            self.pool.append(csi)

    def cluster(self):
        while True:
            try:
                with self.lock:
                    data = self.pool.popleft()
                y = self.model(data)
                print(y)
            except:
                pass

    def run(self):
        t = threading.Thread(target=self.cluster)
        t.start()


if __name__ == '__main__':
    path = 'model.pkl'
    cluster = Cluster(path)
    cluster.run()
    # You can use the "put_data" method anytime, anywhere
    cluster.put_data()
