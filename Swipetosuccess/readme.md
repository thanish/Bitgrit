# Swipe to Success

This repo contains all the code including preprocessing, model that I wrote for the Bitgrit's Swipetosuccess competition. 
The goal of this competition is to predict the level of compatibility of two given users to improve the profile recommendation algorithm for yenta which is a professional 
networking app to connect students, professional, investors, startup to collaborate for a successful professional growth.
For this purpose, the predictive model should be able to classify the level of compatibility between user A and user B into 4 categories: 

• No Match = 0: At least one of either user A or user B swiped left on the other, meaning there is no possibility of a match.

• Match = 1: Both user A and user B swiped right on each other and matched.

• Matched and met but unfavorable review = 2: Both user A and user B swiped right on each other and matched, then met. After the meeting, user A gave user B a review of 1-3 out of 5 (an “unfavorable” review).

• Matched and met and favorable review = 3: Both user A and user B swiped right on each other and matched, then met. After the meeting, user A gave user B a review of 4-5 out of 5 (a “favorable” review).

To build the model there were 2 different types of data subsets provided: user data and interaction data.
I. User data: These files are connected through the user_id column
II. Interaction data: These files are indexed by from-to user_id pairs (e.g. 12345-52462)
III. Train and test files: In order to train the model, we provide a train.csv file with pairs of user IDs and their corresponding scores
