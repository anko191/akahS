from django.urls import path
from . import views
urlpatterns = [
    path('', views.shaka, name='home'),
]
