# Utrecht Klachtenbot 1.0
This bot is designed to demonstrate the various ways in which even simple automatic selection algorithms can interfere with public values. It allows users to file a complaint about the municipality of Utrecht and let the bot analyze its contents. It will analyze on the following metrics:

- Toxicity detection (using [RobBERT-dutch-base-toxic-comments](https://huggingface.co/ml6team/robbert-dutch-base-toxic-comments))
- Topic categorization through keyword matching
- Detection of terms indicating urgency
- Neighborhood of the user

It weighs these metrics to calculate a priority score for each complaint. 

All results are temporarily stored in a table. This is reset upon reloading the bot.

## How to edit parameters
In the Streamlit app, click the arrow on the top left to change certain parameters of this bot, including:

- High priority categories
- Treshold for toxicity detection
- Whether toxic content should be filtered out
- Priority score weight for neighborhoods
- Keywords for each category

## How to run
To run the Streamlit app locally, simply download the file and run in the command line:

``streamlit run Klachtenbot.py``

Alternatively, visit the publicly hosted application on [Streamlit](https://klachtenbot.streamlit.app/).
