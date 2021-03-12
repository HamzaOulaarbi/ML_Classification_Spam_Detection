# ML_Classification_Spam_Detection
ML_Classification_Spam_Detection
## Task Details 
Explore text message data and create models to predict if a message is spam or not.

## Conclusions

Differents algorithms, vectorizers and new features are tested to see the impact on the AUC score. Here are the results with the different configuration:

The AUC score using CountVectorizer and Multinomial Naive Bayes is 0.97.

The AUC score using TfidfVectorizer and Multinomial Naive Bayes is 0.94.

The AUC score using TfidfVectorizer, adding one feature and SVM is 0.97.

The AUC score using TfidfVectorizer, adding two features and Logistic Regression is 0.97.

The AUC score using CountVectorizer, adding three features and Logistic Regression is 0.98

   => The 10 largest coefficients from this model : ['digit_count' 'ia' ' r' 'xt' 'ne' 'co' ' ba' ' x' 'ian ' '46']
   
   => The 10 smallest coefficients from this model :[' i' 'ca' '..' '. ' 'pe' ' go' ' m' 'if' 'us' 'go']
   
The second configuration show that TfidfVectorizer is less robust than the CountVectorizer. However, by  asjusting some parameters and adding more features models even with TfidfVectorizer become more robust
