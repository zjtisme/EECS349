import numpy as np 
import matplotlib.pyplot as plt 
x = [1,2,3,4,5,6,7,8]
models = ['zeroR','DT','K-NN','SVM','NB','MLP','AdaBoost','RandomForest']

accuracy1 = [0.64, 0.6, 0.65, 0.64, 0.63, 0.64, 0.65, 0.63]
accuracy2 = [0.64, 0.78, 0.72, 0.76, 0.62, 0.75, 0.66, 0.81]
# plt.xticks(x, models, rotation=20)
# plt.plot(x, accuracy1,'go--')
# plt.title('Performance of different classifiers without DepTime attribute')
# plt.xlabel('Classifiers')
# plt.ylabel('Highest Accuracy')
# plt.show()

attributes = ['Month','DayofMonth', 'DayofWeek','DepTime', 'CRSDepTime','CRSArrTime','UniqueCarrier','FlightNum','Origin','Dest','Distance']
importance1 = [ 0.06749874 , 0.21832652 , 0.11817252 , 0.10796583 , 0.11708806 , 0.02353097,
  0.1228456  , 0.06317055 , 0.07141834 , 0.08998288]
importance2 = [ 0.05075579 , 0.0954737 ,  0.05347642 , 0.30995766 , 0.23510176 , 0.05554391,
  0.02338392 , 0.05400532 , 0.03716656 , 0.03806188 , 0.04707308]
x = np.arange(len(attributes))

plt.bar(x, importance2, align='center', alpha=0.5)
plt.xticks(x, attributes, rotation=22)
plt.xlabel('Attributes')
plt.ylabel('Importance')
plt.title('Attributes Importance Evaluation - with DepTime')
plt.show()