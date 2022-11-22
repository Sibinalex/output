#Output Solidity


    pragma solidity^0.6;

    contract Student_management
    {
        struct Student { 
            int stud_id;
            string name ;
            string department; 
        }

        Student[] Students;

        function add_stud(int stud_id, string memory name, string memory department )public{

            Student memory stud =Student(stud_id,name,department); 
            Students.push(stud);

        }

        function getStudent(int stud_id) public view returns(string memory,string memory){
            for (uint i=0;i<Students.length;i++){ 
                Student memory stud=Students[i];
                if(stud.stud_id==stud_id){
                return(stud.name, stud.department);
                } 
            }
        return("NotFound","NotFound"); 
        }
    }



ML P1============================================================
             
    df = df.iloc[:,1:]
    df['pickup_datetime'] = pd.to_numeric(pd.to_datetime(df['pickup_datetime']))
    df['key'] = pd.to_numeric(pd.to_datetime(df['key']))
    df.dropna(inplace=True)
    
    x = df.drop('fare_amount', axis=1)
    y = df.iloc[:,1:2]
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
    
    from sklearn.linear_model import LinearRegression
    lrmodel = LinearRegression()
    lrmodel.fit(x_train,y_train)
    
    y_predict = lrmodel.predict(x_test)
    from sklearn.metrics import mean_squared_error
    lrmodelrsme = np.sqrt(mean_squared_error(y_predict,y_test))
    lrmodelrsme
    
    from sklearn.ensemble import RandomForestRegressor
    rfmodel = RandomForestRegressor(n_estimators=100,random_state=101)
    
    y_train = y_train.values.ravel()
    rfmodel.fit(x_train,y_train)
    y_rfrpredict = rfmodel.predict(x_test)
    
    rfr = np.sqrt(mean_squared_error(y_rfrpredict,y_test))
    rfr
    
    
ML_P2========================================================================
 
    import matplotlib.pyplot as plt 
    from sklearn.feature_extraction.text import CountVectorizer 
    # from sklearn.model_selection import GridSearchCV 
    from sklearn import svm
    
    X = data['Email No.'].values
    y = data['ect'].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, Y_test = train_test_split( X, y, test_size=0.8, random_state=0)
    
    cv = CountVectorizer ( )
    X_train = cv.fit_transform(X_train)
    X_test = cv.transform(X_test)

    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 10)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test,Y_test))


    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train,y_train)
    print(knn_model.score(X_test,Y_test))

ML_P4========================================================================    


    import sympy as sy
    x = sy.Symbol('x')
    a = sy.diff((x+5)**2)
    cur_x = 3
    rate = 0.01 
    precision = 0.00001 
    previous_step_size = 1 
    max_iters = 1000
    iters = 0

    # df = lambda x: 2*(x+3)

    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x 
        # cur_x = cur_x - rate * df(prev_x) 
        cur_x = cur_x - rate * a.subs('x',prev_x)
        previous_step_size = abs(cur_x - prev_x) 
        iters = iters+1 
        print("Iteration",iters,"\nX value is",cur_x) 


    print("The local minimum occurs at", cur_x)
    
    
ML_P5========================================================================    

    # for column in df.columns [1:-3]:
    #   df[column].replace(0, np.NaN, inplace = True)
    #   df[column].fillna(round(df[column].mean (skipna=True)), inplace = True)
    
    X = df.iloc[:, :8] #Features
    Y = df.iloc[:, 8:] #Predictor
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn_fit = knn.fit(X_train, Y_train.values.ravel ())
    knn_pred = knn_fit.predict(X_test)
    
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
    print( "Confusion Matrix")
    print(confusion_matrix(Y_test, knn_pred))
    print("Accuracy Score:", accuracy_score(Y_test, knn_pred))
    print("Reacal Score:", recall_score(Y_test, knn_pred))
    print ("F1 Score:", f1_score(Y_test, knn_pred))
    print("Precision Score:",precision_score(Y_test, knn_pred))


    
ML_P6========================================================================    

    import matplotlib.pyplot as plt
    
    X = dataset.iloc[:, [1,2]]. values
    
    wcss = []
    from sklearn.cluster import KMeans
    
    
    for i in range (1,11):
      kmeans = KMeans(n_clusters= i, init= 'k-means++', random_state = 21)
      kmeans.fit(X)
      wcss.append(kmeans.inertia_)

    wcss
    
    
    plt.plot(range(1,11), wcss)
    plt.title('WCSS via Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS Value')
    plt.show()
    
    
    kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
    y_means = kmeans.fit_predict(X)
    
    plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'Yellow', label = 'Cluster 1')
    plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'Red', label = 'Cluster 2')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='blue', label = 'Centroids')
    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income(k$)')
    plt.ylabel('Spending Score(1-100')
    plt.show()
    
========================
