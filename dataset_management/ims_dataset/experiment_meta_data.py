"""
Files include the information for how each separate channel for a given test should be processed as well as what the recorded failure mode was


Values for training data rougly based on the data from this paper by Liu et all:  Domain Adaptation Digital Twin for Rolling Element Bearing Prognostics see table 2


"""

channel_info_test_1 = [
    {
        "measurement_name": "bearing1_channel1",
        "mode": None,
        "healthy_records": [10,100]
    },
    {
        "measurement_name": "bearing1_channel2",
        "mode": None,
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing2_channel1",
        "mode": None,
        "healthy_records": [10, 100]
    },
    {
        "measurement_name": "bearing2_channel2",
        "mode": None,
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing3_channel1",
        "mode": "inner",
        "healthy_records": [10, 100]
    },
    {
        "measurement_name": "bearing3_channel2",
        "mode": "inner",
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing4_channel1",
        "mode": "ball",
        "healthy_records": [10, 100]
    },
    {
        "measurement_name": "bearing4_channel2",
        "mode": "ball",
        "healthy_records": [10, 100]
    },
]

channel_info_test_2 = [
    {
        "measurement_name": "bearing1_channel1",
        "mode": "outer",
        "healthy_records": [10,100]
    },

    {
        "measurement_name": "bearing2_channel2",
        "mode": None,
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing3_channel3",
        "mode": None,
        "healthy_records": [10, 100]
    },

    {
        "measurement_name": "bearing4_channel4",
        "mode": None,
        "healthy_records": [10, 100]
    },
]

channel_info_test_3 = [
    {
        "measurement_name": "bearing1_channel1",
        "mode": None,
        "healthy_records": [11000,100]
    },

    {
        "measurement_name": "bearing2_channel2",
        "mode": None,
        "healthy_records": [11000, 100]
    },

    {
        "measurement_name": "bearing3_channel3",
        "mode": "outer",
        "healthy_records": [11000, 100]
    },

    {
        "measurement_name": "bearing4_channel4",
        "mode": None,
        "healthy_records": [11000, 100]
    },
]
