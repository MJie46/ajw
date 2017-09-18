"""learn_models URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib import staticfiles
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve


from people import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^index/', views.index),
    url(r'^insertGuestData/', views.insertGuestData),
    url(r'^seatSearch/', views.seatSearch),
    url(r'^seatDetail/', views.seatDetail),
    url(r'^testdb$', views.testdb),
    url(r'^testanimte', views.testanimte),
    url(r'^static/(?P<path>.*)$',serve,{"document_root":settings.STATIC_ROOT}),

]

# url(r'^static/(?P<path>.*)', 'django.views.static.serve', {'document_root': 'd:/wwwsite/office/static'}),
# + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)