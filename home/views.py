
from django.shortcuts import render

# Create your views here.

from joblib import load
import torch
import torch.nn as nn
from .DNN import dense_model
from .CNN import CNN_model
import numpy as np
model = load('model.joblib')
scaler=load('scaler.joblib')

def home(request):
   
    return render(request, 'home.html')
def res(request):
    age=int(request.GET['age'])
    gen=int(request.GET['gender'])
    hdi=int(request.GET['heartdisease'])
    hpt=int(request.GET['hypertension'])
    mar=int(request.GET['maritalstatus'])
    wor=int(request.GET['worktype'])
    bmi=float(request.GET['bmi'])
    smoke=int(request.GET['smoking'])
    res=int(request.GET['residence'])
    glu=float(request.GET['glu'])
    stroke=int(request.GET['stroke'])
    al=int(request.GET['alcohol'])
    pa=int(request.GET['physicalactivity'])
    stress=float(request.GET['stress'])
    HDL=int(request.GET['HDL'])
    LDL=int(request.GET['LDL'])
    UBP=int(request.GET['UBP'])
    LBP=int(request.GET['LBP'])
    HR=int(request.GET['HR'])
    CP=int(request.GET['CP'])
    Dz=int(request.GET['Dz'])
    sc=scaler.transform([[age,glu,bmi,stress,LBP,UBP,HDL,LDL,HR]])
    data=[[sc[0][0],gen,hpt,hdi,mar,wor,res,sc[0][1],sc[0][2],smoke,al,pa,stroke,sc[0][3],sc[0][6],sc[0][7],sc[0][4],sc[0][5],sc[0][8],CP,Dz]]
    lr_probs=model.predict_proba(data)[:, 1]
    pred1=model.predict(data)
    data1 = torch.tensor(data, dtype=torch.float32)
    test_outputs = dense_model(data1)
    DNN_probs= torch.softmax(test_outputs, dim=1)[:, 1].detach().numpy()
    test_preds = torch.argmax(test_outputs, dim=1)
    test_outputs_CNN = CNN_model(data1)
    CNN_probs = torch.softmax(test_outputs_CNN, dim=1)[:, 1].detach().numpy()
    test_preds_CNN = torch.argmax(test_outputs_CNN, dim=1)
    
    DNN_CONFIDENCE_RANGE = (0.4, 0.6)
    model_probs = {
            'lr': lr_probs,
            'dnn': DNN_probs,
            'cnn': CNN_probs
        }
    risk_assessment = calculate_risk_assessment(model_probs)
    
    return render(request,'res.html',{'risk_assessment': risk_assessment})
def calculate_risk_assessment(model_probs):
    RISK_THRESHOLDS = {
        'LOW': 0.3,
        'MEDIUM': 0.6,
        'HIGH': 0.8
    }
    lr_prob = float(model_probs['lr'].item()) if isinstance(model_probs['lr'], np.ndarray) else float(model_probs['lr'])
    dnn_prob = float(model_probs['dnn'].item()) if isinstance(model_probs['dnn'], np.ndarray) else float(model_probs['dnn'])
    cnn_prob = float(model_probs['cnn'].item()) if isinstance(model_probs['cnn'], np.ndarray) else float(model_probs['cnn'])
    
    # Calculate ensemble probability
    ensemble_prob = (lr_prob + dnn_prob + cnn_prob) / 3
    
    if ensemble_prob < RISK_THRESHOLDS['LOW']:
        risk_level = 'LOW'
        color = 'success'
    elif ensemble_prob < RISK_THRESHOLDS['MEDIUM']:
        risk_level = 'MEDIUM'
        color = 'warning'
    else:
        risk_level = 'HIGH'
        color = 'danger'
    
    # Generate explanations and recommendations
    explanations = {
        'LOW': "Your results indicate a low probability of heart disease.",
        'MEDIUM': "Some risk factors are present that warrant attention.",
        'HIGH': "Multiple risk factors indicate elevated heart disease risk."
    }
    
    def make_clickable(url, text=None):
        """Helper function to create clickable HTML links"""
        display_text = text if text else url.split('?')[0].replace('https://youtu.be/', '')
        return f'<a href="{url}" target="_blank" style="color: #1a73e8; text-decoration: underline;">{display_text}</a>'
    
    recommendations = {
        'LOW': {
            'precautions': [
                "Check blood pressure, cholesterol, and blood sugar regularly",
                "Get 7-9 hours of good-quality sleep each night",
                "Control blood pressure through diet, exercise, and medication if needed",
                "Maintain a healthy weight (especially reduce belly fat)",
                "Stay physically active (30+ minutes of moderate exercise daily)"
            ],
            'food': {
                'diet': [
                    "Focus on: Olive oil, fish, leafy greens, whole grains, tomatoes, beans, nuts, seeds, berries",
                    "Cut back on saturated fats, trans fats, salt, sugar, and processed foods"
                ],
                'videos': [
                    make_clickable("https://youtu.be/zRhGJAWswwE", "10 Vegetables to Clean Arteries"),
                    make_clickable("https://youtu.be/3OAqdNYeyAA", "Best Foods for Heart Health"),
                    make_clickable("https://youtu.be/MwxiRaeE_bQ", "10 Foods That Reduce Heart Attack Risk")
                ]
            },
            'exercise': [
                make_clickable("https://youtu.be/dj03_VDetdw", "Brisk Walking Routine"),
                make_clickable("https://youtu.be/fPinEXphaq0", "Swimming Exercises"),
                make_clickable("https://youtu.be/z7PGuInGMZ4", "Cycling Workout"),
                make_clickable("https://youtu.be/-KB_phT0VCE", "Full Body Exercise")
            ],
            'stress_management': [
                "Practice meditation, deep breathing, yoga, or hobbies",
                "Talk to someone you trust about stress"
            ]
        },
        'MEDIUM': {
            'precautions': [
                "Quit smoking and alcohol completely",
                "Limit saturated/trans fats, red meat, processed foods, and sugary drinks",
                "Follow a Mediterranean-style diet",
                "Avoid excess salt in meals",
                "Get 7-9 hours of sleep nightly",
                "Stay properly hydrated",
                "Schedule regular health checkups",
                "Do daily walks (30+ minutes)",
                "Avoid high sugar content"
            ],
            'food': {
                'diet': [
                    "Mediterranean diet recommended",
                    "Focus on plant-based foods and healthy fats"
                ],
                'videos': [
                    make_clickable("https://youtu.be/RiK7lCSfj4s", "Heart-Healthy Meals"),
                    make_clickable("https://youtu.be/XfIK5NjgjwU", "Diet After Stent Placement"),
                    make_clickable("https://youtu.be/iVuptvRieuE", "Nutrition Guide")
                ]
            },
            'exercise': [
                make_clickable("https://youtu.be/d9WTYDuZpFQ", "Low-Impact Cardio"),
                make_clickable("https://youtu.be/X2nqJ4UMnU8", "Daily Walking Routine"),
                make_clickable("https://youtu.be/V1A9wv_Hw7Y", "Beginner Exercises")
            ]
        },
        'HIGH': {
            'urgent': [
                '<span style="color: #d32f2f; font-weight: bold;">CONSULT A DOCTOR IMMEDIATELY - YOU ARE IN DANGER</span>',
                "Requires professional medical intervention"
            ],
            'actions': [
                "Follow all medium-risk precautions under medical supervision",
                "Immediate lifestyle changes required",
                "Strict medication adherence if prescribed"
            ]
        }
    }
    
    
    return {
        'level': risk_level,
        'score': round(ensemble_prob * 100, 1),
        'color': color,
        'explanation': explanations[risk_level],
        'recommendations': recommendations[risk_level],
        'model_breakdown': {
            'Logistic Regression': model_probs['lr'] * 100,
            'Deep Neural Network': model_probs['dnn'] * 100,
            'Convolutional NN': model_probs['cnn'] * 100,
            'Ensemble Average': ensemble_prob * 100
        }
    }