from django.contrib import admin
from django.urls import path, include
from api import views

urlpatterns = [
    path('predict/', views.predict, name="predict"),
    path('pdf/', views.sendPDF, name="sendPDF"),
    path('feedback/', views.save_feedback, name="save_feedback"),
]