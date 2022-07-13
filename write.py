import numpy as np
import pandas as pd
import csv
import io
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
def run_algo1(file_path):

    diabetes = pd.read_csv('/Users/xiazhijie/ECE 157/Lab1/diabetes.csv')
    diabetes.head()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    training_accuracy = []
    test_accuracy = []
    # try n_neighbors from 1 to 10
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
    # build the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

    df=pd.read_csv(file_path, low_memory=False, skipinitialspace=True, na_filter=False)
    df["Outcome"]=" "
    features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    Xnew=df[features].values 
    y_predict=knn.predict(Xnew)
    
    #get the figure plot
    fig = plt.figure()
    pos = plt.scatter(Xnew[:,1],Xnew[:,2], y_predict == 1,cmap='jet')
    neg = plt.scatter(Xnew[:,1],Xnew[:,2], y_predict == 0)
    plt.xlabel('Glucose')
    plt.ylabel('Blood Pressure')
    plt.legend((pos,neg),('Positive', 'Negative'))
    plt.title('Classification Results on Diabetes with KNN')
    return mpld3.fig_to_html(fig)


#run_algo1('unknowns.csv')
#plt.show()

def run_algo2(file_path):
    def load_data(filename):
        with open(filename,'r') as file:
            reader = csv.reader(file)
            columnNames = next(reader)
            rows = np.array(list(reader), dtype = float)
            return columnNames, rows
    
    def seperate_labels(columnNames, rows):
        labelColumnIndex = columnNames.index('quality')
        ys = rows[:,labelColumnIndex]
        xs = np.delete(rows,[1, 4, 6, 7, 11],axis=1)
        del columnNames[labelColumnIndex]
        return columnNames, xs, ys
    def preposess(columnNames, rows):
        xs = np.delete(rows,[1,4,6,7],axis=1)
        return xs
    from sklearn.ensemble import RandomForestClassifier
    def rtf(X_train, X_test, y_train):
    
        gnb = RandomForestClassifier(n_estimators = 580)
        gnb.fit(X_train,y_train)
        y_pred=gnb.predict(X_test)
        return y_pred


    columnNames,X_test =load_data(file_path)
    x_test=preposess(columnNames, X_test)
    columnNames,data =load_data('/Users/xiazhijie/ECE 157/Lab2/white_wine.csv')
    columnNames, xs, ys = seperate_labels(columnNames, data)
    result = rtf(xs,x_test,ys)
    fig, ax = plt.subplots()
    pms = plt.scatter(x_test[:,0],x_test[:,1],10, result, cmap = 'jet')
    plt.xlabel('fixed acidity')
    plt.ylabel('citric acid')
    cbar = fig.colorbar(pms, ax=ax)
    plt.title('Regression Results on Wine Data with Random Forest Classifier')
    cbar.set_label('Wine Quality')
    pms.set_clim(3, 9)

    
    return mpld3.fig_to_html(fig)

#run_algo2('unknowns2.csv')
#plt.show()


def run_algo3(file_path):

    import csv
    import numpy as np
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(file_path)
    df = data.drop(['Player','Pos','Tm'],axis=1) # drop columns with strings
    df = df.dropna(axis=1) #drop col with missing values
    df.head()


    selected_features=['TRB','AST','PTS']
    def load_clean_normed_data():
        df = pd.read_csv('/Users/xiazhijie/ECE 157/Lab3/nba_players_stats_19_20_per_game.csv')[['Player']+selected_features]
        for stat in selected_features:
            df[stat] = df[stat]/df[stat].max() #Normalize
        return df

    df = load_clean_normed_data()

    def train_one_class_svm(data):
        from sklearn.svm import OneClassSVM
        return OneClassSVM(kernel = 'rbf').fit(data[selected_features])

    def train_elliptic_envelope(data):
        from sklearn.covariance import EllipticEnvelope
        return EllipticEnvelope(contamination = 0.05,random_state = 42).fit(data[selected_features])

    def train_isolation_forest(data):
        from sklearn.ensemble import IsolationForest
        return IsolationForest(contamination = 0.05,random_state = 42).fit(data[selected_features])
    clf_ee = train_elliptic_envelope(df)
    scores = clf_ee.decision_function(df[selected_features])
    topthreeIndices = np.argsort(scores)[:3] #gives indices of top3 , sorted list of scores
    top3 = df.iloc[topthreeIndices] # iloc: gives me samples of these indices

    outliers = scores < 0 #find outliers
    inliers = scores >= 0 #find inliers
    outliersSansTopThree = outliers.copy()
    outliersSansTopThree[topthreeIndices] = False

    fig = plt.figure()

    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(df.iloc[~outliers][selected_features[0]],
            df.iloc[~outliers][selected_features[1]],
            df.iloc[~outliers][selected_features[2]],label = 'Inliers')
    ax.scatter(df.iloc[outliersSansTopThree][selected_features[0]],
            df.iloc[outliersSansTopThree][selected_features[1]],
            df.iloc[outliersSansTopThree][selected_features[2]],label = 'Outliers')
    ax.scatter(top3[selected_features[0]],
            top3[selected_features[1]],
            top3[selected_features[2]], label='Top three outliers')

    ax.legend()
    ax.set_xlabel('TRB')
    ax.set_ylabel('AST')
    ax.set_zlabel('PTS')
    #plt.legend((inliers,outliers, top3),('Inliers', 'Outliers', 'Top three outliers'))
    plt.title('Outliers Detection on NBA Data with Elliptic Envelop')

    # with io.StringIO() as stringbuffer:
    #     fig.savefig(stringbuffer, format = 'svg')
    #     svgstring = stringbuffer.getvalue()
    # return svgstring

    with io.StringIO() as stringbuffer:
        fig.savefig(stringbuffer,format='svg')
        svgstring = stringbuffer.getvalue()
    return svgstring

#run_algo3('nba_players_stats_19_20_per_game.csv')
#plt.show()
