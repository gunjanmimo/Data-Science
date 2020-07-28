from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("nba_logreg.csv")

# missing value checking
data.drop('3P%', axis=1, inplace=True)


# we do not need the name column
X = data.iloc[:, 1:19].values
y = data.iloc[:, 19].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ANN

bclfr = Sequential()

bclfr.add(Dense(units=10, kernel_initializer='uniform',
                activation='relu', input_dim=18))
bclfr.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
bclfr.add(Dense(units=10, kernel_initializer='uniform', activation='sigmoid'))
bclfr.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
bclfr.fit(X_train, y_train, batch_size=10, epochs=100)


y_pred = bclfr.predict(X_test)
