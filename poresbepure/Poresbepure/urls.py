
from . import views 
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.home, name='home'),
    path('identify/', views.identify, name='identify'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('identifyskin/', views.identifyskin, name='identifyskin'),
    path('skinguide/', views.skinguide, name='skinguide'),
    path('skintips/', views.skintips, name='skintips'),
    path('identify/result', views.result, name='result'),]
    