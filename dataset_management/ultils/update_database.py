import itertools
import time

from tqdm import tqdm

from database_definitions import make_db
from multiprocessing import Pool


def new_derived_doc(query, source_name, target_name, function_to_apply,db_to_act_on):
    # Loop through all the documents that satisfy the conditions of the query

    db, client = make_db(db_to_act_on)

    source_collection = db[source_name]
    target_collection = db[target_name]

    if source_collection.count_documents(query) == 0:
        print("No examples match the query in the source database")

    for doc in source_collection.find(query):
        computed = function_to_apply(doc, db)  # TODO: Need keyword arguments to make this work. Or global variable?

        # Create a new document for each of the computed features, duplicate some of the original data
        for feature in computed:  # TODO: Could make use of insert_many?
            # TODO: Figure out how to deal with overwrites

            new_doc = {"mode": doc["mode"],
                       "severity": doc["severity"],
                       "meta_data": doc["meta_data"],
                       "augmented": doc["augmented"]
                       }

            new_doc.update(feature)  # Add the newly computed data to a document containing the original meta data

            target_collection.insert_one(new_doc)

    client.close()
    return target_collection


class DerivedDoc():
    def __init__(self, query, source_name, target_name, function_to_apply,db_to_act_on,chunk_size=100):
        self.query = query
        self.source_name = source_name
        self.target_name = target_name
        self.process = function_to_apply

        # Instantiate the databases related to this process
        self.db, self.client = make_db(db_to_act_on)
        self.source_collection = self.db[source_name]
        self.target_collection = self.db[target_name]

        self.chunk_size = chunk_size
        self.cursor = self.source_collection.find(self.query, batch_size = self.chunk_size)

        # self.process_arguments = (doc for doc in self.source_collection.find(self.query))
        # TODO: Need to work with batches so that everything can fit into ram

    def parallel_do(self,process_arguments):
        pool = Pool()
        result = pool.map(self.process, process_arguments)
        return result

    def serial_do(self, process_arguments):
        return [self.process(arg) for arg in tqdm(process_arguments)]

    def update_database(self, parallel=True):
        # TODO: It might be that the idea of not updating the database separately in each process, could lead to memmory issues.
        # An alternative would be to do the updates inside the parallel process.

        docs_at_start = self.target_collection.count_documents({})
        t_start = time.time()

        # chunks = yield_rows(cursor, CHUNK_SIZE)
        # for chunk in chunks:
        #     # do processing here
        #     pass

        for chunck in self.yield_rows():
            process_arguments =  (doc for doc in chunck)
            if parallel:
                result = self.parallel_do(process_arguments)
            else:
                result = self.serial_do(process_arguments)

            # This will be a list of list (of documents). We flatten them to insert them into the database
            flattened = itertools.chain.from_iterable(result)
            self.target_collection.insert_many(flattened)

        docs_at_end = self.target_collection.estimated_document_count()
        print("Time elapsed applying {}: ".format(self.process.__name__), time.time() - t_start, "sec")
        print("Parallel :{}".format(str(parallel)))
        print("roughly {} documents added".format(docs_at_end - docs_at_start))
        print("")

    def yield_rows(self):
        """
        Generator to yield chunks from cursor

        Adapted from answer here:
       https://stackoverflow.com/questions/54815892/pymongo-cursor-batch-size 
        """
        chunk = []
        for i, row in enumerate(self.cursor):
            if i % self.chunk_size == 0 and i > 0:
                yield chunk
                del chunk[:]
            chunk.append(row)
        yield chunk



def new_docs_from_computed(doc, list_of_computed_dicts):
    # Transfer some of information from the source document

    updated_dicts = []
    for computed_dict in list_of_computed_dicts:
        # new_doc = {"mode": doc["mode"],
        #            "severity": doc["severity"],
        #            "meta_data": doc["meta_data"],
        #            "augmented": doc["augmented"]
        #            }

        to_transfer = ["mode", "severity", "meta_data", "augmented"]

        new_doc = {key: doc[key] for key in to_transfer}
        # new_doc = doc["mode", "severity"]

        new_doc.update(computed_dict)  # Add the newly computed data to a document containing the original meta data
        updated_dicts.append(new_doc)
    return updated_dicts

# return [new_doc.copy().update(computed_dict) for computed_dict in list_of_computed_dicts]
