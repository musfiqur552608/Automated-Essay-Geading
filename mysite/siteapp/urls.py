from os import name
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='index'),
    path('site', views.site, name='site'),
]