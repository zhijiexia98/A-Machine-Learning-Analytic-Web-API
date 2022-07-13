from BONBON import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = 'BONBON'
urlpatterns = [
    path('', views.YourViewName.as_view(), name='BONBON') ,
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)