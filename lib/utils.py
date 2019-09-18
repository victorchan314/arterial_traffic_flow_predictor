import sys

class Tee(object):
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.file.flush()
