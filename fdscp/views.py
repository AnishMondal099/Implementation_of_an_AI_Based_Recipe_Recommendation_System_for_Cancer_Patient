from django.http import HttpResponse
from django.shortcuts import render
from service.models import Service
from django.contrib.auth.hashers import make_password, check_password
from django.shortcuts import render
import pandas as pd
from django.shortcuts import render
from fdscp.project import preprocess_data, build_neural_network, recommend_recipes
from django.views.decorators.csrf import csrf_exempt
def login(request):
    return render(request,"index.html")
def register(request):
    return render(request,"register.html")
def savedata(request):
    msg=""
    if request.method=="POST":
        name=request.POST.get('name')
        email=request.POST.get('email')
        password=request.POST.get('password')
        cpassword=request.POST.get('cpassword')
        if password != cpassword:
            msg="Password not same"
            return render(request,"register.html",{"msg":msg})
        else:
            try:
                hashed_text=make_password(password)
                data=Service(name=name,email=email,password=hashed_text)
                data.save()
                msg="Data saved"
                return render(request,"register.html",{"msg":msg})
            except:
                msg="Email already exist"
                return render(request,"register.html",{"msg":msg})

def login_view(request):
    msg=""
    if request.method=="POST":
        email=request.POST.get('email')
        password=request.POST.get('password')
        try:
          data=Service.objects.filter(email=email)
          for record in data:
            enpass=record.password
          is_correct = check_password(password,enpass)
          if is_correct==False:
              msg="Email or Password is wrong"
              return render(request,"index.html",{"msg":msg})
          else:
               return render(request,"loginview.html")

        except:
            msg="Email or Password is wrong"
            return render(request,"index.html",{"msg":msg})
# def foods(request):
#     def foods(request):
#         # output = predict_recommendation({"some": "input"})  # adjust input as needed
#         return render(request, "foods.html",)
# Load and preprocess dataset once
df = pd.read_excel("cancer_recipe_recommendation_dataset.xlsx")
df, feature_matrix, encoder, scaler = preprocess_data(df)
model = build_neural_network(input_dim=feature_matrix.shape[1])
model.fit(feature_matrix, [1] * len(feature_matrix))  # Dummy training

@csrf_exempt
def recommend_view(request):
    if request.method == 'POST':
        profile = {
            key: request.POST[key] for key in request.POST
        }

        # Convert numeric values
        for k in ['Caloric_Intake_Requirement (kcal/day)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Fiber (g)', 'Cooking_Time (minutes)','Fat_Limit (g/day)']:
            profile[k] = float(profile[k])

        recommendations = recommend_recipes(profile, df, feature_matrix, encoder, scaler, model)
        food_list = [f"{row['Cuisine']} - {row['Preparation_Method']} - {row['Caloric_Intake_Requirement (kcal/day)']} kcal" for _, row in recommendations.iterrows()]

        return render(request, 'foods.html', {'food_list': food_list})
           

