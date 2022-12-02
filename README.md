# Moodlist - Fully Connected Multi-Mood Classifier Based On Recent Music Streaming History

What if the music you listened to was influenced by how you were feeling at the moment? 
We wanted to introduce a new heuristic to increase personalization of music recommendation, by analyzing a person’s mood based on their recent listening history. 

# Contents
- [Data](#the-data)
  * [The Dataset](#the-dataset)
  * [Visualization](#visualization)
  * [Data Cleaning](#data-cleaning)
- [Data Processing](#data-processing)
- [Models](#models)
  * [Random Forest Classifier](#random-forest-classifier)
  * [Fully Classified Network](#fully-connected-network)
- [Results](#results)

# The Data

## The Dataset

The team used the ”Spotify Million Playlists Dataset” as our training data along with the Spotify API which provides the model parameter data for each song. The four parameters (Acousticness, Valence, Energy, Danceability) are fed into the model.
## Visualization

We compared acousticness, danceability and valence across the 5 mood classifications. There were
some clear distinctions between the moods, with ”angry” having a generally low danceability and
low acousticness, ”calm” having a lower danceability, ”happy” being skewed towards higher dance-
ability and in general higher valence, ”love” having a high valence and ”sad” having a notably low
valence. These indicate patterns in our data and prove to be promising for our models success.

## Data Cleaning


# Data Processing

To generate a uniform dataset, the team split the data into input sets of 9 songs each. Thus, each input is represented by a .CSV file with the cleaned data from the 9 songs. Once completed, we re-iterated the data preparation steps to confirm the uniformity of the data. The inputs are organized into folders for the mood they represent. In the volume of inputs, there's a lot of variation. This would incorporate bias towards specific moods during the training of the model. To fix this issue, the team leveled out the number of inputs so that each mood would have the same proportion of data, with 2000 inputs for each mood.

# Models
 
## Random Forest Classifier

## Fully Classified Network

# Results
