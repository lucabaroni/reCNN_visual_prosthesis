import pickle
def pickleread(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    return x

def picklesave(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)

