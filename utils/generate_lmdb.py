import lmdb
import os

from tqdm import tqdm
from utils.others import TimeCounter


_10TB = 10995116277760
_1TB = 1099511627776
_1GB = 1073741824


# Get length of lmdb
def get_length(lmdb_dir):
	env = lmdb.open(lmdb_dir, readonly=True, map_size=_1GB)
	operator = env.begin()

	length = int(operator.get('length'.encode()).decode())

	env.close()
	return length


# dump dict to lmdb
def dump_lmdb(data_dict, lmdb_dir, verbose=True):
	os.makedirs(lmdb_dir, exist_ok=True)

	# open lmdb
	env = lmdb.open(lmdb_dir, map_size=_10TB)

	with env.begin(write=True) as operator:
		if verbose:
			iter_dict = tqdm(data_dict.items(), desc="Dumping data...")
		else:
			iter_dict = data_dict.items()

		for k, v in iter_dict:
			operator.put(key=str(k).encode(), value=str(v).encode())

	env.close()


# Dump jsonl to lmdb
def jsonl2lmdb(jsonl_path, lmdb_dir):
	os.makedirs(lmdb_dir, exist_ok=True)

	# open lmdb
	env = lmdb.open(lmdb_dir, map_size=_10TB)

	with env.begin(write=True) as operator:
		with TimeCounter("Loading data..."):
			with open(jsonl_path, 'r', encoding='utf-8') as r:
				cnt = 0
				for line in tqdm(r, desc="Parsing jsonl..."):
					operator.put(key=str(cnt).encode(), value=line.encode())
					cnt += 1

		info = "Keys are as follows:\n" \
			   "	info: decription of dataset\n" \
			   "	length: length of data\n" \
			   "	0 ~ length-1: index of each data\n"

		operator.put(key='info'.encode(), value=info.encode())
		operator.put(key='length'.encode(), value=str(cnt).encode())

	env.close()