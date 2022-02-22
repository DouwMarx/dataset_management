from database_definitions import make_db

def new_derived_doc(query, source_name, target_name, function_to_apply):
    # Loop through all the documents that satisfy the conditions of the query

    db, client = make_db()

    source_collection = db[source_name]
    target_collection = db[target_name]

    # processed = db["processed"]
    # augmented = db["augmented"]
    # encoding = db["encoding"]
    # metrics = db["metrics"]


    cursor =  source_collection.find(query)

    # print(cursor.count())
    # if cursor.count() == 0:
    if source_collection.count_documents(query)==0:
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

