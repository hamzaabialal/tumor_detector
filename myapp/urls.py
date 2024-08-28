"""
URL configuration for djangoProject2 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    path('elements/', views.ElementsPageView.as_view(), name='elements_page'),
    path('index/', views.IndexPageView.as_view(), name='home_page'),
    path('tumor/', views.TumorDetectionView.as_view(), name='index'),
    path('', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
]