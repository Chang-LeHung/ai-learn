from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size
        mini_batch = random.sample(self.buffer, batch_size)
        return zip(*mini_batch)

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    data: deque = deque(maxlen=3)
    data.append((1, 1))
    data.append((3, 3))
    data.append((2, 2))
    data.append((5, 5))
    print(data)
    # n1, n2 = zip(*data)
    # print(f"{n1 = } {n2 = }")
