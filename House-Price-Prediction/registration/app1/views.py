from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np

# Using GridSearchCV to find the best algorithm for this problem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

import csv
from django.shortcuts import render

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd



df=pd.read_csv(r"C:\Users\lenovo\Desktop\Django_Login_System-main\registration\app1\cleaned_data.csv",index_col=False)


# Create your views here.
@login_required(login_url='login')
def HomePage(request):
        # Read CSV file and extract values for dropdown
    csv_file_path = r'C:\Users\lenovo\Desktop\Django_Login_System-main\registration\app1\cleaned_data.csv'

    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file_path,index_col=False)

    # Extract values from the 5th column for dropdown
    dropdown_values = df.iloc[:, 4].unique()  # Extract values from the 5th column (index 4)

    context = {
        'dropdown_values': dropdown_values
    }
    return render (request,'home.html', context)

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('login')
        



    return render (request,'signup.html')




def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('login')


def showlist(request):
    return render(request, "show.html")

def result(request):
    df['bhk'] = df['bhk'].astype(float)

    X=df.drop(columns=['price'])
    y=df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    numeric_features = ['bhk', 'bath','total_sqft']
    categorical_features = ['location']

    numeric_transformer = StandardScaler(with_mean=False)
    categorical_transformer = OneHotEncoder(drop='first', sparse=False)
    column_trans = make_column_transformer((OneHotEncoder(),['location']))

    # Define the ColumnTransformer
    preprocessor = ColumnTransformer(
    transformers=[
        ('location', OneHotEncoder(), ['location']),
        ('bhk', StandardScaler(with_mean=False), ['bhk']),
        ('bath', StandardScaler(with_mean=False), ['bath']),
        ('total_sqft', StandardScaler(with_mean=False), ['total_sqft']),
    ],
    remainder='passthrough'
    )


    #scaler = StandardScaler(with_mean=False)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #column_trans=ColumnTransformer(transformers=[('num',scaler,['location'])])
    #column_trans = make_column_transform
    # []'price_per_sqft', [0]],er(transformers=(OneHotEncoder(),['location']),remainder='passthrough')
    #pipe = make_pipeline(preprocessor, column_trans,LinearRegression())
    pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])
    pipe.fit(X_train,y_train)

    

    location = request.GET.get('location')
    bhk = float(request.GET.get('bhk'))
    bath = float(request.GET.get('bath'))
    total_sqft = float(request.GET.get('total_sqft'))

    print(location,bhk,bath,total_sqft)

    user_input = pd.DataFrame({
    'location': [location],
    'bhk': [bhk],
    'bath': [bath],
    'total_sqft': [total_sqft],
    })

        

    context = [
        ['Location', location],
        ['BHK', bhk],
        ['Bath', bath],
        ['price_per_sqft', [0]],
        ['Total Sqft', total_sqft]
    ]

    #X_transformed = pipe.fit_transform(user_input)
    feature=['price']
    target = np.array(df[feature].values)
    #prediction = pipe.predict(input)[0]*1e5
    #return str(np.round(prediction,2))

    y_pred_lr=pipe.predict(user_input)[0]

    price=("Predicted price is rs "+str(np.round(y_pred_lr,2)))

    return render(request,'home.html',{"result2":price})


import csv
from django.shortcuts import render

def dropdown_view(request):
    # Path to your CSV file

    # Read CSV file and extract values for dropdown
    with open(df):
        reader = csv.reader(df)
        dropdown_values = [row[4] for row in reader]  # Assuming the first column contains the values

    context = {
        'dropdown_values': dropdown_values
    }

    return render(request, 'home.html', context)
