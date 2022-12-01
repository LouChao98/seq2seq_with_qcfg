import hashlib
import os
MD5_BUF_SIZE = 65536


def get_hash(files, **kwargs):
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python

    md5 = hashlib.md5()
    for file in files:
        with open(file, "rb") as f:
            while True:
                data = f.read(MD5_BUF_SIZE)
                if not data:
                    break
                md5.update(data)

    md5_args = hashlib.md5(str(kwargs).encode(encoding="utf-8"))
    return md5.hexdigest() + "_" + md5_args.hexdigest()

def iter_dir(path):
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            yield os.path.join(dirpath, filename)