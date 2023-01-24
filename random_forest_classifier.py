from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = datasets.load_breast_cancer()

X = data.data
Y = data.target

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=50)

sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

classifier = RandomForestClassifier(n_estimators=3, criterion='entropy', random_state=0)
classifier.fit(X_Train, Y_Train)

Y_Pred = classifier.predict(X_Test)
Y_Pred2 = classifier.predict(X_Train)

cm = confusion_matrix(Y_Test, Y_Pred)
cm2 = confusion_matrix(Y_Train, Y_Pred2)

acc1 = accuracy_score(Y_Test, Y_Pred)
acc2 = accuracy_score(Y_Train, Y_Pred2)


