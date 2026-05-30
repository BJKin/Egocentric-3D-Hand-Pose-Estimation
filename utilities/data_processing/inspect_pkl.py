"""
Reports total entries and extracts unique video UIDs in order to download specific Ego4D videos.
"""

import re
import pickle

data = pickle.load(open(r"..\..\data\ego4d_hands\grasp_ego.pkl", "rb"))
print(f"Total entries: {len(data)}")

uid_re = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
uids = sorted({uid_re.search(k).group(0) for k in data.keys() if uid_re.search(k)})
open("kitchen_uids.txt","w").write("\n".join(uids))

    