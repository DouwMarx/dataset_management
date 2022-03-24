from database_definitions import make_db
db_ims,client_ims = make_db("ims_test2_bearing1_channel1")
db_ims_rapid,client_ims_rapid = make_db("ims_rapid0_test2_bearing1_channel1")

# db_ims_test,client_ims_test = make_db()

db_phenomenological_rapid,client_phenomenological_rapid = make_db("phenomenological_rapid0")
db_phenomenological,client_phenomenological = make_db("phenomenological")
