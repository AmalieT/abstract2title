# abstract2title

Predict the title of a materials science paper from its abstract using the magic of a bidirectional LSTM.
To download/preprocess data (this takes a LONG time):
./download_and_preprocess.bash

Use abstract2title.py to train

Expects the following environment variables to be defined;
MATSCHOLAR_STAGING_HOST
MATSCHOLAR_STAGING_USER
MATSCHOLAR_STAGING_PASS
MATSCHOLAR_STAGING_DB