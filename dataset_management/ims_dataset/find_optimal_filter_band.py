"""
This project is initially interested in using the envelope spectrum

For this we need the optimal frequency band for impulsive information.

We use the Kurtogram to get this frequency band.

This script searches for the optimal frequency band for the IMS dataset.
"""

from database_definitions import make_db

db,client = make_db("ims")
