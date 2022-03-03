from database_definitions import make_db
import pickle
from pathlib import Path

output_path = Path("/home/douwm/projects/PhD/code/signal_processing/data")

db, client = make_db("phenomenological_rapid")

sample = db["raw"].find_one({"severity":"2"}, {'_id': False}) # ID removed from results

fname = output_path.joinpath("example_sample.pkl")


# pickle.dumps(fname)
with open(fname, 'wb') as file:
    # pickle.dump(pickle.loads(sample["time_series"]), file)
    pickle.dump(sample, file)
