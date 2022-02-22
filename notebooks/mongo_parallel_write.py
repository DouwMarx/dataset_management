from database_definitions import processed,raw
from dataset_management.general.update_database_with_processed import compute_features_from_time_series_doc
from dataset_management.ultils.update_database import new_derived_doc
from multiprocessing import Pool

processed.delete_many({})

source_collection = raw

# Process the time data
query = {"time_series": {"$exists": True}}
# new_derived_doc(query, "raw", "processed", compute_features_from_time_series_doc)
# print(processed.find_one({}).keys())

# The "process" will be compute features from time series doc
parallel_process = compute_features_from_time_series_doc


# the arguments for these function are doc and db : DB can be removed if we make a class of it?

from database_definitions import db
function_arguments = ((doc,db) for doc in source_collection.find(query))

# for i in function_arguments:
#     compute_features_from_time_series_doc(*i)
#     print("i")

pool = Pool()  # Create a multiprocessing Pool
r = pool.starmap(parallel_process,function_arguments)
